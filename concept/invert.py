# Original source code: https://github.com/huggingface/peft
# The code is taken from examples/lora_dreambooth/train_dreambooth.py and performs LoRA Dreambooth
# finetuning, it was modified for integration in the Material Palette pipeline. It includes some
# minor modifications but is heavily refactored and commented to make it more digestible and clear.
# It is rather self-contained but avoids being +1000 lines long! The code has two interfaces:
# the original CLI and a functinal interface via `invert()`, they have the same default parameters!

import os
import math
import itertools
from pathlib import Path
from argparse import Namespace

import torch
from tqdm.auto import tqdm
import torch.utils.checkpoint
import torch.nn.functional as F
from accelerate.utils import set_seed
from diffusers.utils import check_min_version

from concept.args import parse_args
from concept.utils import load_models, load_optimizer, load_logger, load_scheduler, load_dataloader, save_lora

# Will throw error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


def main(args):
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    set_seed(args.seed)

    root_dir = Path(args.data_dir)
    assert root_dir.is_dir()

    output_dir = args.prompt.replace(' ', '_')
    output_dir = root_dir.parent.parent / 'weights' / root_dir.name / output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    ckpt_path = output_dir / f'checkpoint-{args.max_train_steps}' / 'text_encoder'
    if ckpt_path.is_dir():
        print(f'{ckpt_path} already exists')
        return ckpt_path.parent

    if args.validation_prompt is not None:
        output_dir_val = output_dir/'val'
        output_dir_val.mkdir()

    ## Load dataset (earliest as possible to anticipate crashes)
    train_dataset, train_dataloader = load_dataloader(args, root_dir)

    from accelerate import Accelerator # avoid preloading before directory validation
    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=output_dir / args.logging_dir,
    )

    logger = load_logger(args, accelerator)

    ## Load scheduler and models
    noise_scheduler, text_encoder, vae, unet = load_models(args)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer, args.learning_rate = load_optimizer(args, unet, text_encoder, accelerator.num_processes)

    lr_scheduler = load_scheduler(args, optimizer)

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Initialize the trackers we use and store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes

    ##! remove args.num_train_epochs and args.gradient_accumulation_steps from CLI
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // len(train_dataloader)
        resume_step = global_step % len(train_dataloader)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, (img, prompt) in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                if args.report_to == "wandb":
                    accelerator.print(progress_bar)
                continue

            # Embed the images to latent space and apply scale factor
            latents = vae.encode(img.to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample a random timestep for each image
            T = noise_scheduler.config.num_train_timesteps
            timesteps = torch.randint(0, T, (len(latents),), device=latents.device, dtype=torch.long)

            # Forward diffusion process: add noise to the latents according to the noise magnitude
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(prompt)[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # L2 error reconstruction objective
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Backward pass on denoiser and optionnally text encoder
            accelerator.backward(loss)

            # Gradient clipping step
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters())
                    if args.train_text_encoder
                    else unet.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            # Handle optimzer and learning rate scheduler
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.report_to == "wandb":
                    accelerator.print(progress_bar)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        _text_encoder = text_encoder if args.train_text_encoder else None
                        save_lora(accelerator, unet, _text_encoder, output_dir, global_step)

            # Log loss and learning rates
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Validation step
            if (args.validation_prompt is not None) and (global_step % args.validation_steps == 0):
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # Create pipeline for validation pass
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    safety_checker=None,
                    revision=args.revision,
                    local_files_only=True)

                # Set `keep_fp32_wrapper` to True because we do not want to remove
                # mixed precision hooks while we are still training
                pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # Set sampler generator seed
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                # Run inference
                for i in range(args.num_validation_images):
                    image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                    image.save(output_dir / 'val' / f'{global_step}_{i}.png')

                del pipeline
                torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _text_encoder = text_encoder if args.train_text_encoder else None
        save_lora(accelerator, unet, _text_encoder, output_dir, global_step)

    accelerator.end_training()
    return ckpt_path.parent

DEFAULT_PROMPT = "an object with azertyuiop texture"
def invert(data_dir: str, prompt=DEFAULT_PROMPT, train_text_encoder=True, gradient_checkpointing=True, **kwargs) -> Path:
    """
    Functional interface for the inversion step of the method. It adopts the same interface as
    the CLI defined in `args.py` by `parse_args` (jump there for details). If the region has already
    been inverted the function will exit early. Always returns the path of the inversion checkpoint.

    :param str `data_dir`: path of the directory containing the region crops to invert
    :param str `prompt`: prompt used for inversion containing the rare token eg. "an object with zkjefb texture"
    :return Path: the path to the inversion checkpoint
    """
    all_args = parse_args(return_defaults=True)
    all_args.update(data_dir=str(data_dir),
                    prompt=prompt,
                    train_text_encoder=train_text_encoder,
                    gradient_checkpointing=gradient_checkpointing,
                    **kwargs)
    return main(Namespace(**all_args))

if __name__ == "__main__":
    args = parse_args()
    args.train_text_encoder = True
    args.gradient_checkpointing = True
    main(args)
