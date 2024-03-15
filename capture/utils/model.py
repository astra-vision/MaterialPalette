import torch.nn as nn

from pathlib import Path

from ..source import ResnetEncoder, MultiHeadDecoder, DenseMTL, DenseReg, Vanilla


def replace_batchnorm_(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.InstanceNorm2d(child.num_features))
        else:
            replace_batchnorm_(child)

def get_model(archi):
    assert archi == 'densemtl'

    encoder = ResnetEncoder(num_layers=101, pretrained=True, in_channels=3)
    decoder = MultiHeadDecoder(
        num_ch_enc=encoder.num_ch_enc,
        tasks=dict(albedo=3, roughness=1, normals=2),
        return_feats=False,
        use_skips=True)

    model = nn.Sequential(encoder, decoder)
    replace_batchnorm_(model)
    return model

def get_module(args):
    loss = DenseReg(**args.loss)
    model = get_model(args.archi)

    weights = args.load_weights_from
    if weights:
        assert weights.is_file()
        return Vanilla.load_from_checkpoint(str(weights), model=model, loss=loss, strict=False, **args.routine)

    return Vanilla(model, loss, **args.routine)

def get_inference_module(pt):
    assert Path(pt).exists()
    model = get_model('densemtl')
    return Vanilla.load_from_checkpoint(str(pt), model=model, strict=False)