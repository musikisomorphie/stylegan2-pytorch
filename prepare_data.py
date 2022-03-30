import numpy as np
import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
import pandas as pd
from pathlib import Path


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    # ../rxrx19b{64, 128, ..., 1024}/images/HUVEC-1/Plate{n}/**.png
    if 'rxrx19b' in file:
        # rxrx19b256
        out, cmp = [], None
        res = file.split('/')[-5]
        for size in sizes:
            file_res = file.replace(res, 'rxrx19b{}'.format(size))
            img = Image.open(file_res).convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format="jpeg", quality=100)
            out.append(buffer.getvalue())

            # only suitable for 128, 256 cases
            debug = False
            if debug:
                if size == 256:
                    im0 = img.crop((0, 0, 256, 256)).resize((128, 128))
                    im0 = np.asarray(im0)
                    im1 = img.crop((256, 0, 512, 256)).resize((128, 128))
                    im1 = np.asarray(im1)
                    ims = np.concatenate((im0, im1), axis=1).astype(np.float32)
                    err = np.abs(ims - cmp)
                    # empirical thres
                    if np.mean(err) > 0.2:
                        print(file, np.max(err), np.mean(err))
                else:
                    # np.float32 avoid integer overflow
                    cmp = np.asarray(img).astype(np.float32)
    else:
        img = Image.open(file)
        img = img.convert("RGB")
        out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out


def prepare(
    env, dataset, trn_imgs, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    files = sorted(dataset.imgs, key=lambda x: x[0])
    # print(len(files))
    if trn_imgs is not None:
        files = list(filter(lambda d: d[0] in trn_imgs, files))
    # print(len(files))
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess images for model training")
    parser.add_argument("--out", type=str,
                        help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="128,256,512,1024",
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))
    path = Path(args.path)
    if 'rxrx1' in args.path:
        if 'rxrx19b' not in args.path:
            csv_raw = pd.read_csv(path / 'metadata.csv')
            csv_raw = csv_raw[csv_raw['site_id'].str.endswith('_1')]
            csv = csv_raw.loc[csv_raw['dataset'] == 'train']
            trn_imgs = dict()
            for row in csv.iterrows():
                r = row[1]
                im_path = path / 'images' / \
                    r.experiment / \
                    'Plate{}'.format(r.plate) / '{}_s1.png'.format(r.well)
                assert im_path not in trn_imgs
                trn_imgs[str(im_path)] = None
        else:
            trn_imgs = None
        img_dir = 'images'
    elif 'scrc' in args.path:
        # be cautious of /
        t_reg = args.out[-4:-1]
        csv_raw = pd.read_csv(path / 'metadata{}.csv'.format(t_reg))
        csv = csv_raw.loc[csv_raw['dataset'] == 'train']
        trn_imgs = dict()
        for row in csv.iterrows():
            r = row[1]
            im_path = path / 'images' / str(r.tma_reg) / \
                '{}_1.png'.format(r.tma_id)
            assert im_path not in trn_imgs
            trn_imgs[str(im_path)] = None
        img_dir = 'images'
    elif 'camelyon17' in args.path:
        trn_imgs = None
        img_dir = 'patches'

    imgset = datasets.ImageFolder(str(path / img_dir))
    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, trn_imgs, args.n_worker,
                sizes=sizes, resample=resample)
