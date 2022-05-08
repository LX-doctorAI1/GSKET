import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset, KnowledgeDataset


class ToPILImage(object):
    def __init__(self):
        self.to_pil = transforms.ToPILImage(mode="F")

    def __call__(self, x):
        return (self.to_pil(x[0]))


class LADataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        normalize = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                         std=[0.275, 0.275, 0.275])
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomCrop(args.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), fillcolor=(0, 0, 0)),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                normalize])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        elif self.dataset_name == 'mimic_cxr':
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            raise NotImplementedError

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'pin_memory': False
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        # image采用默认的batch方式，多进程的情况下可以减少额外的内存拷贝
        images = default_collate([d['image'] for d in data])

        images_id = [d['idx'] for d in data]
        reports_ids = [d['report_ids'] for d in data]
        reports_masks = [d['report_mask'] for d in data]
        seq_lengths = [d['seq_length'] for d in data]
        knowledges_ids = [d['knowledge_ids'] for d in data]
        # knowledges_masks = [d['knowledge_mask'] for d in data]

        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        b_knowledges = np.zeros((len(knowledges_ids), 64), dtype=int)
        # b_knowledges_masks = np.zeros((len(reports_ids), 64), dtype=int)

        for i, knowledge_ids in enumerate(knowledges_ids):
            knowledge_ids = knowledge_ids[:64]
            b_knowledges[i, :len(knowledge_ids)] = knowledge_ids

        # for i, knowledge_mask in enumerate(knowledges_masks):
        #     knowledge_mask = knowledge_mask[:64]
        #     b_knowledges_masks[i, :len(knowledge_mask)] = knowledge_mask

        batch = {'images_id': images_id,
                 'images': images,
                 'targets': torch.LongTensor(targets),
                 'reports_mask': torch.FloatTensor(targets_masks),
                 'knowledges': torch.LongTensor(b_knowledges),
                 # 'knowledges_mask': torch.FloatTensor(b_knowledges_masks)
                 }
        return batch


def get_bert_dataloader(args, tokenizer):
    dataset = KnowledgeDataset(args, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
    return dataset, dataloader

