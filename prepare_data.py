import lmdb
import argparse
import multiprocessing

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from functools import partial
from torchvision import datasets


def resize_and_convert(img, size):
    if img.size != (size, size):
        img = img.resize((size, size))
    buffer = BytesIO()
    # use lossless png format
    img.save(buffer, format='png')
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size))

    return imgs


def resize_worker(img_file, sizes):
    i, file = img_file
    # ../rxrx19b{64, 128, ..., 1024}/images/HUVEC-1/Plate{n}/**.png
    # if 'rxrx19b' in file:
    #     # rxrx19b256
    #     out, cmp = [], None
    #     res = file.split('/')[-5]
    #     for size in sizes:
    #         file_res = file.replace(res, 'rxrx19b{}'.format(size))
    #         img = Image.open(file_res).convert('RGB')
    #         buffer = BytesIO()
    #         img.save(buffer, format='png')
    #         out.append(buffer.getvalue())

    #         # only suitable for 128, 256 cases
    #         debug = False
    #         if debug:
    #             if size == 256:
    #                 im0 = img.crop((0, 0, 256, 256)).resize((128, 128))
    #                 im0 = np.asarray(im0)
    #                 im1 = img.crop((256, 0, 512, 256)).resize((128, 128))
    #                 im1 = np.asarray(im1)
    #                 ims = np.concatenate((im0, im1), axis=1).astype(np.float32)
    #                 err = np.abs(ims - cmp)
    #                 # empirical thres
    #                 if np.mean(err) > 0.2:
    #                     print(file, np.max(err), np.mean(err))
    #             else:
    #                 # np.float32 avoid integer overflow
    #                 cmp = np.asarray(img).astype(np.float32)
    # else:
    img = Image.open(file)
    out = resize_multiple(img, sizes=sizes)

    return i, out


def prepare(env, dataset, n_worker, sizes):
    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    resize_fn = partial(resize_worker, sizes=sizes)
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
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)
    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes)
