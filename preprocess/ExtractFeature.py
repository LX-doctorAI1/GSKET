import sys
sys.path.append('..')
import ipdb
import argparse
import os
import re
import json
from glob import glob
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from skimage.io import imread, imsave
from torchvision import transforms
from tqdm import tqdm
import torchxrayvision as xrv
from misc.utils import seed_everything, set_logging


def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""

    if sample.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(sample.max(), maxval))

    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    # sample = sample / np.std(sample)
    return sample


class MIMIC_Dataset(Dataset):
    """
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S.
    MIMIC-CXR: A large publicly available database of labeled chest radiographs.
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self, imgpath, jsonpath, transform=None, data_aug=None,
                 seed=0, unique_patients=True):
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        with open(jsonpath, 'r') as f:
            ann = json.load(f)
        self.ann = ann['train'] + ann['val'] + ann['test']

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        item = self.ann[idx]

        # subjectid = item['subject_id']
        # studyid = item["study_id"]
        dicom_id = item["id"]

        img_path = os.path.join(self.imgpath, item['image_path'][0])
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        return {"img": img, "dicom_id": dicom_id, "idx": idx}


class CXR_Dataset(Dataset):
    """
    OpenI
    Dina Demner-Fushman, Marc D. Kohli, Marc B. Rosenman, Sonya E. Shooshan, Laritza
    Rodriguez, Sameer Antani, George R. Thoma, and Clement J. McDonald. Preparing a
    collection of radiology examinations for distribution and retrieval. Journal of the American
    Medical Informatics Association, 2016. doi: 10.1093/jamia/ocv080.

    Dataset website:
    https://openi.nlm.nih.gov/faq

    Download images:
    https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d
    """

    def __init__(self, imgpath, jsonpath, transform=None, data_aug=None,
                 seed=0, unique_patients=True):
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        with open(jsonpath, 'r') as f:
            ann = json.load(f)
        self.ann = ann['train'] + ann['val'] + ann['test']

    def __len__(self):
        return len(self.ann)

    def get_image(self, img_path):
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)
        return img

    def __getitem__(self, idx):
        item = self.ann[idx]
        dicom_id = item["id"]
        img = []
        for path in item['image_path']:
            img_path = os.path.join(self.imgpath, path)
            img.append(self.get_image(img_path))

        img = torch.stack(img, dim=0)

        return {"img": img, "dicom_id": dicom_id, "idx": idx}


def load_dataset(cfg):
    data_aug = None
    if cfg.data_aug:
        data_aug = torchvision.transforms.Compose([
            xrv.datasets.ToPILImage(),
            torchvision.transforms.RandomAffine(cfg.data_aug_rot,
                                                translate=(cfg.data_aug_trans, cfg.data_aug_trans),
                                                scale=(1.0 - cfg.data_aug_scale, 1.0 + cfg.data_aug_scale)),
            torchvision.transforms.ToTensor()
        ])
        # print(data_aug)

    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    if cfg.dataset_name == 'mimic':
        dataset = MIMIC_Dataset(
            imgpath=os.path.join(cfg.dataset_dir, "images-224-MIMIC/files"),
            jsonpath=os.path.join(cfg.dataset_dir, 'annotation.json'),
            transform=transforms, data_aug=data_aug, unique_patients=False)
    elif cfg.dataset_name == 'iu':
        dataset = CXR_Dataset(
            imgpath=os.path.join(cfg.dataset_dir, "images/iu_2image/images/"),
            jsonpath=os.path.join(cfg.dataset_dir, 'images/iu_2image/annotation.json'),
            transform=transforms, data_aug=data_aug, unique_patients=False)
    else:
        raise NotImplementedError

    return dataset


def main(cfg):
    os.makedirs(cfg.output, exist_ok=True)
    seed_everything(cfg.seed)
    device = 'cuda' if cfg.cuda else 'cpu'

    model = torchvision.models.resnet101(num_classes=18, pretrained=False)
    # patch for single channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(cfg.weights).state_dict())
    model.to(device)
    model.eval()
    logging.info('Loaded ResNet101 model and weight')

    train_dataset = load_dataset(cfg)
    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads,
                                               pin_memory=cfg.cuda)
    logging.info(f'Loaded dataset, dataset size {len(train_dataset)}')

    # already_save = glob(cfg.output + '/*npy')
    with torch.no_grad():
        data_loader = tqdm(train_loader)
        for batch_idx, samples in enumerate(data_loader):
            dicom_id = samples['dicom_id']
            # if dicom_id + '.npy' in already_save:
            #     continue

            images = samples["img"].to(device)
            outputs = []
            for i in range(images.shape[1]):
                outputs.append(model(images[:, i]).cpu().detach())
            # outputs = F.sigmoid(outputs)

            # bs x 2 x 18
            outputs = torch.stack(outputs, dim=1)
            outputs = torch.mean(outputs, dim=1)
            # BS x 18
            features = outputs.numpy()

            for i in range(features.shape[0]):
                feature = features[i]
                np.save(os.path.join(cfg.output, dicom_id[i] + '.npy'), feature)


if __name__ == "__main__":
    set_logging('extract_feature.log')
    logging.info('Start.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="/home/shuxinyang/data/mimic/xrv",
                        help="the begin directory of all")
    parser.add_argument('--dataset_name', type=str, default="mimic",
                        help="the begin directory of all")
    parser.add_argument('--weights', type=str, default="/home/shuxinyang/code/torchxrayvision/scripts/output/mimic_ch-resnet101-pretrain-best.pt")
    parser.add_argument('-o', '--output', type=str, default="/home/shuxinyang/data/mimic/xrv/features")
    parser.add_argument('-g', '--gpu', type=str, default="5")

    parser.add_argument('--cuda', type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--threads', type=int, default=4, help='')
    parser.add_argument('--data_aug', type=bool, default=True, help='')
    parser.add_argument('--data_aug_rot', type=int, default=45, help='')
    parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    logging.info(str(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info(f'Set GPU ID: {args.gpu}')

    seed_everything(args.seed)
    logging.info(f'Set seed {args.seed} done!')

    try:
        main(args)
    except Exception as e:
        logging.error(e)
    finally:
        logging.info('Done!')
