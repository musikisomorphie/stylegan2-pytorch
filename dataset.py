from io import BytesIO
import cv2
import lmdb
import torch
import sparse
import pickle
import random
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from wilds.datasets.wilds_dataset import WILDSDataset


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
            img = npz['img'][96:-96, 96:-96]
            img = Image.fromarray(img)
            # the resize step is inspired by clean-FID
            img = img.resize((128, 128), resample=Image.Resampling.BICUBIC)
            img = np.asarray(img).clip(0, 255).astype(np.uint8)
            img = self.transform(img)
            rna = npz['key_melanoma_marker']
            rna = torch.from_numpy(rna).to(img).float()
        return img, rna


class STDataset(WILDSDataset):
    def __init__(self,
                 data,
                 gene_num,
                 gene_spa=False,
                 root_dir=Path('Data'),
                 transform=None,
                 split_scheme=None,
                 debug=False,
                 seed=None,
                 # compatible to stylegan3
                 resolution=128,
                 num_channels=3,
                 max_size=None):

        self._dataset_name = data
        self.gene_num = gene_num
        self.gene_spa = gene_spa
        self.img_dir = root_dir / f'{data}/GAN/crop'
        self.trans = transform
        # This is for compatible to stylegan3 training
        self.debug = debug
        if seed:
            random.seed(seed)
        self.resolution = resolution
        self.num_channels = num_channels

        # Prep metadata including cropped image paths
        df = pd.read_csv(str(self.img_dir / 'metadata.csv'))
        self._input_array = df.path.values
        self.ext = self._input_array[0].split('_')[-1]
        print(f'Path extension of {data}: {self.ext}')

        # Prep genedata img if exists
        self.expr_img = self.prep_gene(self.img_dir / 'metadata_img.csv',
                                       True)

        # Prep genedata cell if exists
        self.expr_cell = self.prep_gene(self.img_dir / 'metadata_cell.csv',
                                        False)

        # Prep subsetdata for downstream analysis
        self.prep_split(df, split_scheme)

    def prep_gene(self, gene_pth, load_name=True):
        if not self.debug:
            assert gene_pth.is_file()

        if gene_pth.is_file():
            _expr = pd.read_csv(str(gene_pth),
                                index_col=0)
            if self._dataset_name in ('CosMx', 'Xenium'):
                _expr = _expr.astype(np.int16)
            elif self._dataset_name == 'Visium':
                _expr = _expr.astype(np.float32)
            _expr = _expr.to_numpy()
        else:
            _expr = None
            print(f'{str(gene_pth)} does not exist')

        # list of gene names
        if load_name:
            with open(str(self.img_dir / 'transcripts.pickle'), 'rb') as fp:
                self.gene_name = pickle.load(fp)
            assert self.gene_num == len(self.gene_name)

        return _expr

    def prep_split(self, df, split_scheme):
        if split_scheme is not None:
            t_nam, t_cnt = np.unique(df[split_scheme].values,
                                     return_counts=True)
            # if the counts have repetitive values, then it cannot be used
            # for stratify subset, thus add another _cond_dct
            self._cond_dct = dict(zip(t_nam, t_cnt))
            if len(t_nam) == 2:
                # this is for GAN training when including dichotomy domain label
                self._split_dict = dict(zip(t_nam, [-1, 1]))
            else:
                self._split_dict = dict(zip(t_nam, list(range(len(t_nam)))))
            self._split_array = df[split_scheme].map(self._split_dict).values

            self._metadata_fields = [split_scheme, ]
            self._metadata_array = list(zip(*[self._split_array.tolist()]))
        else:
            # add dummy metadata
            self._metadata_array = [[0] for _ in range(len(self._input_array))]

        # add dummy labels
        self._y_array = torch.ones([len(self._input_array)])
        self._y_size, self._n_classes = 1, 1

    def get_input(self, idx):
        img_pth = self.img_dir / self._input_array[idx]
        gene_pth = str(img_pth).replace(self.ext, 'rna.npz')

        if not self.debug:
            img, gene_expr = self.run_input(img_pth, gene_pth, idx)
        else:
            img = torch.empty([0, 128, 128])
            if self.expr_img is not None:
                # Here img is the processed gene img and gene cell cmp
                img, gene_expr = self.run_debug(
                    self.expr_img[idx], gene_pth, idx)
            else:
                gene_expr = sparse.load_npz(gene_pth)
                # This part is for creating metadata_img.csv
                if self._dataset_name in ('CosMx', 'Xenium'):
                    gene_expr = gene_expr.sum((0, 1)).todense()
                    gene_expr = gene_expr.astype(np.int16)
        return img, gene_expr

    def run_input(self, img_pth, gene_pth, idx):
        img = np.array(Image.open(img_pth))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).contiguous().float()

        if self.gene_spa:
            assert self._dataset_name in ('CosMx', 'Xenium')
            gene_expr = sparse.load_npz(gene_pth)

            if self.trans is not None:
                img, gene_expr == self.trans([img, gene_expr])
            else:
                gene_expr = gene_expr.todense().transpose((2, 0, 1))
                gene_expr = torch.from_numpy(gene_expr).contiguous().float()
        else:
            gene_expr = self.expr_img[idx]
            gene_expr = torch.from_numpy(gene_expr).contiguous().float()

            if self.trans is not None:
                img = self.trans(img)
        return img, gene_expr

    def run_debug(self, gene_expr, gene_pth, idx):
        out = sparse.load_npz(gene_pth).todense()
        if self._dataset_name in ('CosMx', 'Xenium'):
            assert (out.sum((0, 1)) == gene_expr).all()
            if self._dataset_name == 'CosMx':
                cell_pth = gene_pth.replace('rna.npz', 'cell.png')
                cell_np = cv2.imread(cell_pth, flags=cv2.IMREAD_UNCHANGED)
                # dir/Liver1/c_1_10_100_rna.npz, cid = 100
                cid = int(Path(gene_pth).name.split('_')[-2])
                out[cell_np != cid] = 0
            else:
                nucl_pth = gene_pth.replace('rna.npz', 'nucl.npz')
                nucl_coo = sparse.load_npz(nucl_pth).todense()
                cell_pth = gene_pth.replace('rna.npz', 'cell.npz')
                cell_coo = sparse.load_npz(cell_pth).todense()
                # dir/Rep1/*_*_1234_*_*_*_*_*_*_*_*_rna.npz, cid = 1234
                cid = int(Path(gene_pth).name.split('_')[2])
                out[(nucl_coo != cid) & (cell_coo != cid)] = 0
            out = np.abs(out.sum((0, 1)) -
                         self.expr_cell[idx])
        else:
            assert (out == gene_expr).all()
        return out, gene_expr


def random_rotation(x: torch.Tensor) -> torch.Tensor:
    x = torch.rot90(x, random.randint(0, 3), [1, 2])
    return x


transform = transforms.Compose(
    [transforms.Lambda(lambda x: random_rotation(x)),
     transforms.RandomHorizontalFlip()]
)
