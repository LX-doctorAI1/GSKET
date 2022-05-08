#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/9
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: fold2lmdb.py
# copy from https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py

import ipdb
import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string
import json

import lmdb
import pickle
import msgpack
from tqdm import tqdm
import pyarrow as pa

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


class MIMICdataset(data.Dataset):
    """自定义dagaset的基类
    提供公用init方法和抽象函数
    """
    def __init__(self, image_dir, json_file, split='train', transform=None):
        self.image_dir = image_dir
        self.json_report = json_file
        self.split = split
        self.transform = transform

        with open(self.json_report, 'r') as f:
            self.report = json.load(f)[split]

        print('=======>> Load {} dataset {} items <<=========='.format(split, len(self.report)))

    def __len__(self):
        return len(self.report)

    def __getitem__(self, index):
        study = self.report[index]
        idx = study['id']
        path = study['image_path'][0]

        name = os.path.splitext(path)[0] + '.jpg'
        image_path = os.path.join(self.image_dir, name)
        image = raw_reader(image_path)
        # image = Image.open(os.path.join(self.image_dir, name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return idx, image


class MIMICdatasetLMDB(data.Dataset):
    """自定义dagaset的基类
    提供公用init方法和抽象函数
    """
    def __init__(self, db_path, json_file, split='train', transform=None):
        self.db_path = db_path
        self.json_report = json_file
        self.split = split
        self.transform = transform

        with open(self.json_report, 'r') as f:
            self.report = json.load(f)[split]

        print('=======>> Load {} dataset {} items <<=========='.format(split, len(self.report)))

        self.db = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __len__(self):
        return len(self.report)

    def __getitem__(self, index):
        study = self.report[index]
        idx = study['id']

        with self.db.begin(write=False) as txn:
            byteflow = txn.get(idx.encode('ascii'))

        imgbuf = pa.deserialize(byteflow)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return idx, image


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(args, write_frequency=5000, num_workers=16):
    # ipdb.set_trace()
    # 设置LMDB数据库
    lmdb_path = osp.join(args.out, "%s.lmdb" % args.name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    # 数据增强
    normalize = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                     std=[0.275, 0.275, 0.275])
    transform = dict()
    transform['train'] = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), fillcolor=(0, 0, 0)),
            transforms.ToTensor(),
            normalize
    ])
    transform['val'] = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
    ])

    # 初始化数据集
    directory = osp.expanduser(args.image_dir)
    print("Loading dataset from %s" % directory)

    train_dataset = MIMICdataset(args.image_dir, args.json_file, split='train', transform=None)
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, collate_fn=lambda x: x)
    val_dataset = MIMICdataset(args.image_dir, args.json_file, split='val', transform=None)
    val_dataloader = DataLoader(val_dataset, num_workers=num_workers, collate_fn=lambda x: x)
    test_dataset = MIMICdataset(args.image_dir, args.json_file, split='test', transform=None)
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, collate_fn=lambda x: x)

    print(len(train_dataset), len(train_dataloader))
    print(len(val_dataset), len(val_dataloader))
    print(len(test_dataset), len(test_dataloader))

    keys = []

    txn = db.begin(write=True)
    for idx, data in enumerate(tqdm(train_dataloader)):
        path, image = data[0]
        txn.put(u'{}'.format(path).encode('ascii'), dumps_pyarrow((image)))
        keys.append(path)
        if idx % write_frequency == 0:
            # print("[%d/%d]" % (idx, len(train_dataloader)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()

    txn = db.begin(write=True)
    for idx, data in enumerate(tqdm(val_dataloader)):
        path, image = data[0]
        txn.put(u'{}'.format(path).encode('ascii'), dumps_pyarrow(image))
        keys.append(path)
        if idx % write_frequency == 0:
            # print("[%d/%d]" % (idx, len(train_dataloader)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()

    txn = db.begin(write=True)
    for idx, data in enumerate(tqdm(test_dataloader)):
        path, image = data[0]
        txn.put(u'{}'.format(path).encode('ascii'), dumps_pyarrow(image))
        keys.append(path)
        if idx % write_frequency == 0:
            # print("[%d/%d]" % (idx, len(train_dataloader)))
            txn.commit()
            txn = db.begin(write=True)
    txn.commit()

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def test_loader(args):
    train_dataset = MIMICdatasetLMDB(args.image_dir, args.json_file)
    train_dataloader = DataLoader(train_dataset, num_workers=4, collate_fn=lambda x: x)

    ipdb.set_trace()
    for idx, data in enumerate(train_dataloader):
        print(idx, data[0])
        print(data[1].shape)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--image_dir", type=str)
    parser.add_argument("-j", "--json_file", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=20)

    args = parser.parse_args()

    # folder2lmdb(args)
    test_loader(args)