from io import BytesIO

import lmdb
import torch
import sparse
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, chn=-1):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform
        self.path = path
        self.chn = chn

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = np.asarray(img)
        if 'rxrx19' in self.path:
            col = img.shape[1] // 2
            img = np.concatenate((img[:, :col],
                                  img[:, col:]), axis=-1)
            if 'rxrx19a' in self.path:
                img = img[:, :, :-1]

            if self.chn != -1:
                img = np.expand_dims(img[:, :, self.chn], -1)
        img = self.transform(img)

        return img


class SODataset(Dataset):
    def __init__(self, data, gene, path, transform):
        if data == 'CosMx':
            self.ext = 'flo.png'
        elif data == 'Xenium':
            self.ext = 'hne.png'
        elif data == 'Visium':
            self.ext = '.npz'

        self.data = data
        self.gene = gene
        self.paths = list(Path(path).rglob(f'*{self.ext}'))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.data in ('CosMx', 'Xenium'):
            img = Image.open(str(self.paths[index]))
            img = np.array(img)
            img = self.transform(img)
            rna = str(self.paths[index]).replace(self.ext, 'rna.npz')
            rna = sparse.load_npz(rna).sum((0, 1)).todense()
            rna = torch.from_numpy(rna).to(img).float()
            if self.data == 'Xenium':
                rna = rna[:self.gene]
        elif self.data == 'Visium':
            npz = np.load(str(self.paths[index]))
            img = npz['img'][64:-64, 64:-64]
            # img = Image.fromarray(img)
            # # the resize step is inspired by clean-FID
            # img = img.resize((128, 128), resample=Image.Resampling.BICUBIC)
            # img = np.asarray(img).clip(0, 255).astype(np.uint8)
            img = self.transform(img)
            rna = npz['key_melanoma_marker']
            rna = torch.from_numpy(rna).to(img).float()
        return img, rna