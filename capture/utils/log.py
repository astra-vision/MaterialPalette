import csv, re, os

from pytorch_lightning.loggers import TensorBoardLogger


def read_csv(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def append_csv(fname, dicts):
    if isinstance(dicts, dict):
        dicts = [dicts]

    if os.path.isfile(fname):
        dicts = read_csv(fname) + dicts

    write_csv(fname, dicts)

def write_csv(fname, dicts):
    assert len(dicts) > 0
    with open(fname, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dicts[0].keys())
        writer.writeheader()
        for d in dicts:
            writer.writerow(d)

def now():
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_info(weights: str):
    search = re.search(r"(.*)_epoch=(\d+)-step", weights)
    if search:
        name, epoch = search.groups()
        return str(name).split(os.sep)[-1], str(epoch)
    return None, None

def get_matlist(cache_dir, dir):
    with open(cache_dir, 'r') as f:
        content = f.readlines()
    files = [dir/f.strip() for f in content]
    return files

def get_logger(args):
    logger = TensorBoardLogger(save_dir=args.out_dir)
    logger.log_hyperparams(args)
