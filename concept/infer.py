import os
import argparse
from pathlib import Path
import random
from itertools import product
from argparse import Namespace

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torchvision.utils import save_image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Inference code for generating samples from concept.")
    parser.add_argument('path', type=Path, default=None)
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--stitch_mode', type=str, default='wmean', choices=['concat', 'mean', 'wmean'])
    parser.add_argument('--resolution', default=1024, choices=[512, 1024, 2048, 4096, 8192], type=int)
    parser.add_argument('--prompt', type=str, default='p1', choices=['p1', 'p2', 'p3', 'p4'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--renorm', action="store_true", default=False)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    args = parser.parse_args()
    return args

def get_roll(x):
    h, w = x.size(-2), x.size(-1)
    dh, dw = random.randint(0,h), random.randint(0,w)
    return dh, dw

def patch(x, k):
    n, c, h, w = x.shape
    x_ = x.view(-1,k*k,c*h*w).transpose(1,-1) # (n, c*h*w, k*k)
    folded = F.fold(x_, output_size=(h*k,w*k), kernel_size=(h,w), stride=(h,w)) # (n, c, h*k, w*k)
    return folded

def unpatch(x, k, p=0):
    n, c, kh, kw = x.shape
    h, w = (kh-2*p)//k, (kw-2*p)//k
    x_ = F.unfold(x, kernel_size=(h+2*p,w+2*p), stride=(h,w)) # (n, c*[h+2p]*[w+2p], k*k)
    unfolded = x_.transpose(1,2).reshape(-1,c,64+2*p,64+2*p) # (n*k*k, c, h+2p, w+2p)
    return unfolded

def get_kernel(p, device):
    x1, x2 = 512-1, 512+2*p-1
    y1, y2 = 1, 0
    fun = lambda x: (y1-y2)/(x1-x2)*x + (x1*y2-x2*y1)/(x1-x2)
    x = torch.arange(512+2*p, device=device)
    y = fun(x)
    y[:512]=1
    y += y.flip(0)
    y -= 1
    Y = torch.outer(y,y)
    return Y[None][None]

def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default"
):
    from peft import PeftModel, LoraConfig
    from diffusers import StableDiffusionPipeline

    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")

    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_name_or_path,
        torch_dtype=dtype,
        local_files_only=True,
        safety_checker=None,
    ).to(device)

    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

def get_vanilla_sd_pipeline(device='cuda'):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
    )
    pipe.to(device)
    return pipe

@torch.no_grad()
def main(args):
    if args.token is None:
        assert args.path.is_dir()
        # global_step =j f"{args.path.name.split('-')[-1]:0>4}"

        token = 'azertyuiop'
        print(f'loading LoRA with token {token}')
        pipe = get_lora_sd_pipeline(ckpt_dir=Path(args.path))
    else:
        token = args.token
        pipe = get_vanilla_sd_pipeline()
        print(f'picked token={token}')

    v_token = token

    prompt = dict(
        p1='top view realistic texture of {}',
        p2='top view realistic {} texture',
        p3='high resolution realistic {} texture in top view',
        p4='realistic {} texture in top view',
    )[args.prompt]
    print(f'{args.prompt} => {prompt}')
    v_prompt = prompt.replace(' ', '-').format('o')
    prompt = prompt.format(token)

    # negative_prompt = "lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, illustration, painting, drawing, art, sketch"
    negative_prompt = ""
    generator = torch.Generator("cuda").manual_seed(args.seed)
    random.seed(args.seed)

    if args.path is not None:
        outdir = args.path/'outputs'
        print(f'ignoring `args.outdir` and using path {outdir}')
        outdir.mkdir(exist_ok=True)
    else:
        # ckpt_dir
        outdir = args.outdir

    reso = {512: 'hK', 1024: '1K', 2048: '2K', 4096: '4K', 8192: '8K'}[args.resolution]
    fname = outdir/f'{v_token}_{reso}_t{args.num_inference_steps}_{args.stitch_mode}_{v_prompt}_{args.seed}.png'

    if fname.exists():
        print('already exists!')
        return fname
    print(f'preparing for {fname}')

    ################################################################################################
    # Inference code
    ################################################################################################
    k= (args.resolution//512)

    num_images_per_prompt=1
    guidance_scale=7.5
    # guidance_scale=1.0

    callback_steps=1
    cross_attention_kwargs=None
    # clip_skip=None
    num_inference_steps=args.num_inference_steps
    eta=0.0
    guidance_rescale=0.0
    callback=None
    callback_steps=1
    output_type='pil'
    height=None
    width=None
    latents=None
    prompt_embeds=None
    negative_prompt_embeds=None

    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(
        prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt*k*k,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        (batch_size * num_images_per_prompt)*k*k,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs.
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # roll noise
            kx, ky = get_roll(latent_model_input)
            latent_model_input = patch(latent_model_input, k)
            latent_model_input = latent_model_input.roll((kx, ky), dims=(2,3))
            latent_model_input = unpatch(latent_model_input, k)

            # split in two for inference
            noise_pred = []
            chunk_size = len(latent_model_input)//16 or 1
            for latent_chunk, prompt_chunk \
                in zip(latent_model_input.chunk(chunk_size), prompt_embeds.chunk(chunk_size)):
                # predict the noise residual
                res = pipe.unet(latent_chunk, t, encoder_hidden_states=prompt_chunk)
                noise_pred.append(res.sample)
            noise_pred = torch.cat(noise_pred)

            # noise unrolling
            noise_pred = patch(noise_pred, k)
            noise_pred = noise_pred.roll((-kx, -ky), dims=(2,3))
            noise_pred = unpatch(noise_pred, k)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()


    if args.resolution == 512:
        decoded = pipe.vae.decode(patch(latents, k) / pipe.vae.config.scaling_factor)
        decoded = decoded.sample.detach().cpu().double()
        images = pipe.image_processor.postprocess(decoded, output_type='pil', do_denormalize=[True]*len(decoded))
        images[0].save(fname)

    ## stiching part
    if args.stitch_mode == 'concat': # naive concatenation
        # image = pipe.vae.decode(folded / pipe.vae.config.scaling_factor)
        chunk_size = len(latents)//16 or 1
        out = []
        for chunk in latents.chunk(chunk_size):
            image = pipe.vae.decode(chunk / pipe.vae.config.scaling_factor)
            out.append(image.sample.detach().cpu().double())
        out = torch.cat(out)

        images = pipe.image_processor.postprocess(out, output_type='pt', do_denormalize=[True]*len(out))
        save_image(images, fname, nrow=k, padding=0)
        # [img.save(f'{i}.png') for i, img in enumerate(images)]

    elif args.stitch_mode == 'mean': # patch mean blending
        p=1
        folded = patch(latents, k)
        folded_padded = F.pad(folded, pad=(p,p,p,p), mode='circular')
        unfolded_padded = unpatch(folded_padded, k, p)

        chunk_size = len(unfolded_padded)//16 or 1
        image_stack = []
        for chunk in unfolded_padded.chunk(chunk_size):
            image = pipe.vae.decode(chunk / pipe.vae.config.scaling_factor)
            image_stack.append(image.sample)
        image_stack = torch.cat(image_stack)

        lmean = image_stack.mean(dim=(-1,-2), keepdim=True)
        gmean = image_stack.mean(dim=(0,2,3), keepdim=True)
        image_stack = image_stack*gmean/lmean

        # with a naive average stitching, the overlap values (bands) are divided
        s = pipe.vae_scale_factor # 1:8 in pixel space
        tp = 2*s*p # total padding
        image_stack[:,:,:tp,:] /= 2.
        image_stack[:,:,-tp:,:] /= 2.
        image_stack[:,:,:,:tp] /= 2.
        image_stack[:,:,:,-tp:] /= 2.

        # gather values into final tensor
        _, c, hpad, wpad = image_stack.shape
        h, w = hpad-tp, wpad-tp
        out_padded = torch.zeros(batch_size, c, h*k+tp, w*k+tp, device=image_stack.device)
        for i, j in product(range(k), range(k)):
            out_padded[:,:,h*i:w*(i+1)+tp,h*j:w*(j+1)+tp] += image_stack[None,i*k+j]

        # accumulate outer bands to opposite sides:
        hp = s*p # half padding
        out_padded[:,:,-tp:-hp,:] += out_padded[:,:,:hp,:]
        out_padded[:,:,hp:tp,:] += out_padded[:,:,-hp:,:]
        out_padded[:,:,:,-tp:-hp] += out_padded[:,:,:,:hp]
        out_padded[:,:,:,hp:tp] += out_padded[:,:,:,-hp:]

        out = out_padded[:,:,hp:-hp,hp:-hp] # trim
        image, *_ = pipe.image_processor.postprocess(out, output_type='pil', do_denormalize=[True])
        image.save(fname)

    elif args.stitch_mode == 'wmean': # weighted average kernel blending
        p=1
        folded = patch(latents, k)
        folded_padded = F.pad(folded, pad=(p,p,p,p), mode='circular')
        unfolded_padded = unpatch(folded_padded, k, p)

        chunk_size = len(unfolded_padded)//16 or 1
        image_stack = []
        for chunk in unfolded_padded.chunk(chunk_size):
            image = pipe.vae.decode(chunk / pipe.vae.config.scaling_factor)
            image_stack.append(image.sample)
        image_stack = torch.cat(image_stack)

        # lmean = image_stack.mean(dim=(-1,-2), keepdim=True)
        # gmean = image_stack.mean(dim=(0,2,3), keepdim=True)
        # image_stack = image_stack*gmean/lmean

        ## patch blending
        scale = pipe.vae_scale_factor
        tp = 2*scale*p # total padding
        mask = get_kernel(scale*p, image_stack.device) # 1:8 in pixel space
        # import pdb; pdb.set_trace()
        # print(mask.shape)
        image_stack *= mask

        # gather values into final tensor
        _, c, hpad, wpad = image_stack.shape
        h, w = hpad-tp, wpad-tp
        out_padded = torch.zeros(batch_size, c, h*k+tp, w*k+tp, device=image_stack.device)
        for i, j in product(range(k), range(k)):
            out_padded[:,:,h*i:w*(i+1)+tp,h*j:w*(j+1)+tp] += image_stack[None,i*k+j]

        # accumulate outer bands to opposite sides:
        hp = scale*p # half padding
        out_padded[:,:,-tp:-hp,:] += out_padded[:,:,:hp,:]
        out_padded[:,:,hp:tp,:] += out_padded[:,:,-hp:,:]
        out_padded[:,:,:,-tp:-hp] += out_padded[:,:,:,:hp]
        out_padded[:,:,:,hp:tp] += out_padded[:,:,:,-hp:]

        out = out_padded[:,:,hp:-hp,hp:-hp] # trim
        image, *_ = pipe.image_processor.postprocess(out, output_type='pil', do_denormalize=[True])
        image.save(fname)

    if args.renorm:
        from . import renorm
        renorm(fname)

    return fname

def infer(path, outdir=None, stitch_mode='wmean', renorm=False, resolution=1024, seed=1, prompt='p1', num_inference_steps=50):
    return main(Namespace(
        path=path,
        outdir=outdir,
        prompt=prompt,
        token=None,
        renorm=renorm,
        stitch_mode=stitch_mode,
        resolution=resolution,
        seed=seed,
        num_inference_steps=num_inference_steps))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
