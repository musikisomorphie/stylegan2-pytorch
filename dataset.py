from io import BytesIO

import lmdb
import numpy as np
from PIL import Image
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
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

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
            col = 3
            if 'rxrx19a' in self.path:
                img = np.concatenate((img[:, :col],
                                      img[:, col:-1]), axis=-1)
            elif 'rxrx19b' in self.path:
                img = np.concatenate((img[:, :col],
                                      img[:, col:]), axis=-1)
            if self.chn != -1:
                img = np.expand_dims(img[:, :, self.chn], -1)
        img = self.transform(img)

        return img
