import itertools
import logging

import torch
import datasets
import diffusers
import transformers
from transformers import CLIPTextModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from peft import LoraConfig, get_peft_model
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from .data import DreamBoothDataset


def load_models(args):
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    elif args.train_text_encoder and args.use_lora:
        config = LoraConfig(
            r=args.lora_text_encoder_r,
            lora_alpha=args.lora_text_encoder_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_text_encoder_dropout,
            bias=args.lora_text_encoder_bias)
        text_encoder = get_peft_model(text_encoder, config)
        text_encoder.print_trainable_parameters()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision)

    if args.use_lora:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules= ["to_q", "to_v", "query", "value"],
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias)
        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()

    ## advanced options
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # below fails when using lora so commenting it out
        if args.train_text_encoder and not args.use_lora:
            text_encoder.gradient_checkpointing_enable()

    return noise_scheduler, text_encoder, vae, unet


def load_optimizer(args, unet, text_encoder, num_processes):
    if args.scale_lr:
        lr = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_processes
    else:
        lr = args.learning_rate

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    if args.train_text_encoder:
        params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())
    else:
        params_to_optimize = unet.parameters()

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)

    return optimizer, lr


def load_logger(args, accelerator):
    # Load wandb if needed
    if args.report_to == 'wandb':
        import wandb
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger = get_logger(__name__)
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    return logger


def load_scheduler(args, optimizer):
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power)
    return lr_scheduler


def save_lora(accelerator, unet, text_encoder, output_dir, global_step):
    ckpt_dir = output_dir / f'checkpoint-{global_step}'
    ckpt_dir.mkdir(exist_ok=True)

    unwrapped_unet = accelerator.unwrap_model(unet)
    unet_dir = ckpt_dir / 'unet'
    unwrapped_unet.save_pretrained(unet_dir, state_dict=accelerator.get_state_dict(unet))

    if text_encoder:
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        textenc_dir = ckpt_dir / 'text_encoder'
        textenc_state = accelerator.get_state_dict(text_encoder)
        unwrapped_text_encoder.save_pretrained(textenc_dir, state_dict=textenc_state)


def load_dataloader(args, root_dir):
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False)

    train_dataset = DreamBoothDataset(
        data_dir=root_dir,
        prompt=args.prompt,
        tokenizer=tokenizer,
        size=args.resolution)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=1)

    return train_dataset, train_dataloader