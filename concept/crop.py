from PIL import Image
from random import shuffle

import torchvision.transforms.functional as tf


def main(path, patch_sizes=[512, 256, 192, 128, 64], threshold=.99, topk=100):
    assert path.is_dir(), \
        f'the provided path {path} is not a directory or does not exist'

    masks_dir = path/'masks'
    assert masks_dir.is_dir(), \
        f'a /masks subdirectory containing the image masks should be present in {path}'

    files = [x for x in path.iterdir() if x.is_file()]
    assert len(files) == 1, \
        f'the target path {path} should contain a single image file!'

    img_path = files[0]
    print(f'---- processing image "{img_path.name}"')

    out_dir = img_path.parent/'crops'
    out_dir.mkdir(parents=True, exist_ok=True)

    pil_ref = Image.open(img_path).convert('RGB')
    img_shape = (pil_ref.width, pil_ref.height)
    ref = tf.to_tensor(pil_ref)

    k = 0
    masks = sorted(x for x in masks_dir.iterdir())
    print(f'  found {len(masks)} masks...')

    for i, f in enumerate(masks):
        clusterdir = out_dir/f.stem

        if clusterdir.exists():
            k += 1
            continue

        pil_mask = Image.open(f).convert('RGB').resize(img_shape)

        main_bbox = pil_mask.convert('L').point(lambda x: 0 if x == 0 else 1, '1').getbbox()
        x0, y0, *_ = main_bbox

        cropped_mask = tf.to_tensor(pil_mask.crop(main_bbox)) > 0

        mask_d = int(cropped_mask[0].float().sum())
        print(f'  > "{f.stem}" cluster, q={cropped_mask[0].float().mean():.2%}')

        kept_bboxes = []
        kept_scales = []
        for patch_size in patch_sizes:
            stride = patch_size//5
            densities, bboxes = patch_image(cropped_mask, patch_size, stride, x0, y0)

            kept_local_res = []
            for d, b in zip(densities, bboxes):
                if d >= threshold:
                    kept_local_res.append(b)

            shuffle(kept_local_res)
            nb_kept = topk - len(kept_bboxes)
            kept_local_res = kept_local_res[:nb_kept]

            kept_bboxes += kept_local_res
            kept_scales += [patch_size]*len(kept_local_res)

            print(f'    {patch_size}x{patch_size} kept {len(kept_local_res)} patches -> {clusterdir}')

            if len(kept_local_res) > 0: # only take largest scale
                break

        if len(kept_bboxes) < 2:
            print(f'   skipping, only found {len(kept_bboxes)} patches.')
            continue

        clusterdir.mkdir(exist_ok=True)
        for i, (s, b) in enumerate(zip(kept_scales, kept_bboxes)):
            cname = clusterdir/f'{i:0>5}_x{s}.png'
            pil_ref.crop(b).save(cname)

        k += 1

    print(f'---- kept {k}/{len(masks)} crops.')

    return out_dir

def patch_image(mask, psize, stride, x0, y0):
    densities, bboxes = [], []
    height, width = mask.shape[-2:]
    for j in range(0, height - psize + 1, stride):
        for i in range(0, width - psize + 1, stride):
            patch = mask[0, j:j+psize, i:i+psize]
            density = patch.float().mean().item()
            densities.append(density)
            bbox = x0+i, y0+j, x0+i+psize, y0+j+psize
            bboxes.append(bbox)
    return densities, bboxes