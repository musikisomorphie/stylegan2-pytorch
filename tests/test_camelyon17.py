import lmdb
import argparse
import numpy as np

from PIL import Image
from io import BytesIO
from pathlib import Path
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import functional as trans_fn


def abserr(a, b):
    print(a.shape, b.shape)
    return np.mean(np.abs(a - b).flatten())


class TestCamelyon17(Dataset):
    def __init__(self, imgs_dir, lmdb_dir, resolution=128):
        self.lmdb_dir = lmdb_dir
        dataset = datasets.ImageFolder(imgs_dir)
        self.img = sorted(dataset.imgs,
                          key=lambda x: x[0])
        self.img = [im for (im, _) in self.img]

        self.env = lmdb.open(
            lmdb_dir,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_dir)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution

        img_ids = np.random.permutation(self.length)
        print(self.length, len(self.img))
        for i in range(10):
            self.__getitem__(img_ids[i])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        print(key)
        bufr = BytesIO()
        imgr = Image.open(self.img[index])
        imgr = imgr.convert('RGB')
        imgr.save(bufr, format="jpeg", quality=100)
        imgr = Image.open(bufr).convert('RGB')
        imgr.save(str(Path(self.lmdb_dir) / '{}_raw.jpg'.format(str(index))))

        bufl = BytesIO(img_bytes)
        imgl = Image.open(bufl).convert('RGB')
        imgl = trans_fn.resize(imgr, 96, Image.LANCZOS)
        imgl.save(str(Path(self.lmdb_dir) / '{}_lmdb.jpg'.format(str(index))))

        print(abserr(np.asarray(imgr),
                     np.asarray(imgl)))


def main():
    parser = argparse.ArgumentParser(
        description='Compute Eigenvalue Distributions')

    parser.add_argument('--imgs-dir',
                        type=str,
                        metavar='DIR',
                        default='/home/jwu/Data/camelyon17_v1.0/patches',
                        help='Path to the image folder')

    parser.add_argument('--lmdb-dir',
                        type=str,
                        metavar='DIR',
                        default='/home/jwu/Data/lmdb/camelyon17',
                        help='Path to the lmdb folder')

    args = parser.parse_args()
    TestCamelyon17(args.imgs_dir, args.lmdb_dir)


if __name__ == '__main__':
    main()
