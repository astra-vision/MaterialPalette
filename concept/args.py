import os
from argparse import ArgumentParser


def get_argparse_defaults(parser):
    # https://stackoverflow.com/questions/44542605/python-how-to-get-all-default-values-from-argparse
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults

def parse_args(return_defaults=False):
    parser = ArgumentParser()

    parser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5',
        help='Path to pretrained model or model identifier from huggingface.co/models.')

    parser.add_argument('--revision', type=str, default=None, required=False,
        help='Revision of pretrained model identifier from huggingface.co/models.')

    parser.add_argument('--seed', type=int, default=1,
        help='A seed for reproducible training.')

    parser.add_argument('--local_rank', type=int, default=-1,
        help='For distributed training: local_rank')


    ## Dataset
    parser.add_argument('--path', type=str, required=True,
        help='A folder containing the training data of instance images.')

    parser.add_argument('--prompt', type=str, default='an object with azertyuiop texture',
        help='The prompt with identifier specifying the instance')

    parser.add_argument('--tokenizer_name', type=str, default=None,
        help='Pretrained tokenizer name or path if not the same as model_name')

    parser.add_argument('--resolution', type=int, default=256, # 512
        help='Resolution of train/validation images')


    ## LoRA options
    parser.add_argument('--use_lora', action='store_true', default=True, # overwrite
        help='Whether to use Lora for parameter efficient tuning')

    parser.add_argument('--lora_r', type=int, default=16, # 8
        help='Lora rank, only used if use_lora is True')

    parser.add_argument('--lora_alpha', type=int, default=27, # 32
        help='Lora alpha, only used if use_lora is True')

    parser.add_argument('--lora_dropout', type=float, default=0.0,
        help='Lora dropout, only used if use_lora is True')

    parser.add_argument('--lora_bias', type=str, default='none',
        help='Bias type for Lora: ["none", "all", "lora_only"], only used if use_lora is True')

    parser.add_argument('--lora_text_encoder_r', type=int, default=16, # 8
        help='Lora rank for text encoder, only used if `use_lora` & `train_text_encoder` are True')

    parser.add_argument('--lora_text_encoder_alpha', type=int, default=17, # 32
        help='Lora alpha for text encoder, only used if `use_lora` & `train_text_encoder` are True')

    parser.add_argument('--lora_text_encoder_dropout', type=float, default=0.0,
        help='Lora dropout for text encoder, only used if `use_lora` & `train_text_encoder` are True')

    parser.add_argument('--lora_text_encoder_bias', type=str, default='none',
        help='Bias type for Lora: ["none", "all", "lora_only"] when `use_lora` & `train_text_encoder` are True')


    ## Training hyperparameters
    parser.add_argument('--train_text_encoder', action='store_true',
        help='Whether to train the text encoder')

    parser.add_argument('--train_batch_size', type=int, default=1,
        help='Batch size (per device) for the training dataloader.')

    # parser.add_argument('--num_train_epochs', type=int, default=1,
    #     help="Number of training epochs, used when `max_train_steps` is not set.")

    parser.add_argument('--max_train_steps', type=int, default=800,
        help='Total number of training steps to perform.')

    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #     help='Number of updates steps to accumulate before performing a backward/update pass.')

    parser.add_argument('--gradient_checkpointing', action='store_true',
        help='Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.')

    parser.add_argument('--scale_lr', action='store_true', default=False,
        help='Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.')

    parser.add_argument('--lr_scheduler', type=str, default='constant',
        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
        help='The scheduler type to use.')

    parser.add_argument('--lr_warmup_steps', type=int, default=0, # 500
        help='Number of steps for the warmup in the lr scheduler.')

    parser.add_argument('--lr_num_cycles', type=int, default=1,
        help='Number of hard resets of the lr in cosine_with_restarts scheduler.')

    parser.add_argument('--lr_power', type=float, default=1.0,
        help='Power factor of the polynomial scheduler.')

    parser.add_argument('--adam_beta1', type=float, default=0.9,
        help='The beta1 parameter for the Adam optimizer.')

    parser.add_argument('--adam_beta2', type=float, default=0.999,
        help='The beta2 parameter for the Adam optimizer.')

    parser.add_argument('--adam_weight_decay', type=float, default=1e-2,
        help='Weight decay to use.')

    parser.add_argument('--adam_epsilon', type=float, default=1e-08,
        help='Epsilon value for the Adam optimizer')

    parser.add_argument('--max_grad_norm', default=1.0, type=float,
        help='Max gradient norm.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
        help='Initial learning rate (after the potential warmup period) to use.')


    ## Prior preservation loss
    # parser.add_argument('--with_prior_preservation', default=False, action='store_true',
    #     help='Flag to add prior preservation loss.')

    # parser.add_argument('--prior_loss_weight', type=float, default=1.0,
    #     help='The weight of prior preservation loss.')

    # parser.add_argument('--class_data_dir', type=str, default=None, required=False,
    #     help='A folder containing the training data of class images.')

    # parser.add_argument('--class_prompt', type=str, default=None,
    #     help='The prompt to specify images in the same class as provided instance images.')

    # parser.add_argument('--num_class_images', type=int, default=100,
    #     help='Min number for prior preservation loss, if lower, more images will be sampled with `class_prompt`.')

    # parser.add_argument('--prior_generation_precision', type=str, default=None,
    #     choices=['no', 'fp32', 'fp16', 'bf16'],
    #     help='Precision type for prior generation (bf16 requires PyTorch>= 1.10 + Nvidia Ampere GPU)')

    ## Logs
    parser.add_argument('--checkpointing_steps', type=int, default=800,
        help='Save a checkpoint every X steps, can be used to resume training w/ `--resume_from_checkpoint`.')

    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
        help='Resume from checkpoint obtained w/ `--checkpointing_steps`, or `"latest"`.')

    parser.add_argument('--validation_prompt', type=str, default=None,
        help='A prompt that is used during validation to verify that the model is learning.')

    parser.add_argument('--num_validation_images', type=int, default=4,
        help='Number of images that should be generated during validation with `validation_prompt`.')

    parser.add_argument('--validation_steps', type=int, default=100,
        help='Run validation every X steps: runs w/ prompt `args.validation_prompt` `args.num_validation_images` times.')

    # parser.add_argument('--output_dir', type=Path, default=None,
    #     help='The output directory where the model predictions and checkpoints will be written.')

    parser.add_argument('--logging_dir', type=str, default='logs',
        help='TensorBoard log directory, defaults default to `output_dir`/runs/**CURRENT_DATETIME_HOSTNAME***.')

    parser.add_argument('--report_to', type=str, default='tensorboard',
        choices=['tensorboard', 'wandb', 'comet_ml', 'all'],
        help='The integration to report the results and logs to.')

    parser.add_argument('--wandb_key', type=str, default=None,
        help='If report to option is set to wandb, api-key for wandb used for login to wandb.')

    parser.add_argument('--wandb_project_name', type=str, default=None,
        help='If report to option is set to wandb, project name in wandb for log tracking.')


    ## Advanced options
    parser.add_argument('--use_8bit_adam', action='store_true',
        help='Whether or not to use 8-bit Adam from bitsandbytes.')

    parser.add_argument('--allow_tf32', action='store_true',
        help='Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.')

    parser.add_argument('--mixed_precision', type=str, default='fp16',
        choices=['no', 'fp16', 'bf16'],
        help='Precision type (bf16 requires PyTorch>= 1.10 + Nvidia Ampere GPU)')

    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true',
        help='Whether or not to use xformers.')

    if return_defaults:
        return get_argparse_defaults(parser)

    args = parser.parse_args()

    env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args