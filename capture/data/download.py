import requests
import argparse
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('outdir', type=Path)
args = parser.parse_args()

url = 'https://ambientcg.com/get?file={}_2K-PNG.zip'

print(f'saving files into {args.outdir}')
args.outdir.mkdir(parents=True, exist_ok=True)

cwd = Path(__file__).parent/'matlist/ambientcg'
with open(cwd.resolve(), 'r') as file:
    materials = [line.strip() for line in file]

mtypes = ['Color', 'Roughness', 'NormalGL']
for uid in tqdm(materials):
    maps = [f'{uid}_2K-PNG_{x}.png' for x in mtypes]

    if all((args.outdir/m).exists() for m in maps):
        continue

    link = url.format(uid)
    try:
        r = requests.get(link)
        archive = ZipFile(BytesIO(r.content))
    except:
        print(f'{uid},{link}\n')
        continue

    for m in maps:
        archive.extract(m, args.outdir)