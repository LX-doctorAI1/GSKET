import os
import json
import pandas as pd
import ipdb
import logging
from tqdm import tqdm

import torch
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset

import six
import pyarrow as pa
from misc.utils import normalize


class BaseDataset(Dataset):
    LOAD_ORDER = ["ANATOMY", "ANATOMY_MODIFIER", "OBSERVATION", "OBSERVATION_MODIFIER"]

    """ann.json文件格式
           {'train': [
               {   "id": "CXR2384_IM-0942",
                   "report": "The heart size and pulmonary vascularity appear within normal limits",
                   "image_path": ["CXR2384_IM-0942/0.png","CXR2384_IM-0942/1.png"],
                   "split": "train",
                   "entity": {
                       "ANATOMY": ["heart"],
                       "OBSERVATION": [],
                       "OBSERVATION_MODIFIER": [],
                       "ANATOMY_MODIFIER": []},
                   "history": ["CXR2888_IM-1290-0001"] }, ...]}
    """

    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.json_report
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.idx2entity = self.load_idx2entity(self.ann)
        self.entity2id = json.loads(open(args.knowledge_file).read())['entity2id']

        self.examples = self.ann[self.split]
        for i in tqdm(range(len(self.examples))):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

            # 遍历所有历史(10个)
            eid = []
            for his in self.examples[i]['history']:
                # 每个历史报告中的所有实体(不定长)
                # if his not in self.idx2entity:
                #     logging.info(f'Not find {his} in id2entity!')
                #     entities = self.idx2entity["CXR1_1_IM-0001"]
                # else:
                #     entities = self.idx2entity[his]

                try:
                    entities = self.idx2entity[his]
                except:
                    entities = self.idx2entity["CXR1_1_IM-0001"]

                for en in entities:
                    if en in self.entity2id:
                        eid.append(self.entity2id[en])
            self.examples[i]['knowledge_ids'] = list(set(eid))
            self.examples[i]['knowledge_mask'] = [1] * len(self.examples[i]['knowledge_ids'])


        logging.info('=======>> Load {} dataset {} items <<=========='.format(split, len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def load_idx2entity(self, ann):
        # 从训练集加载id和报告中实体的映射关系
        # return: {idx: [entity1, entity2, ...]}
        ann = ann['train'] + ann['val'] + ann['test']
        idx2entity = {}
        for item in ann:
            idx = item['id']
            entity = item['entity']

            idx2entity[idx] = []
            for o in self.LOAD_ORDER:
                idx2entity[idx].extend(entity[o])

        return idx2entity


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        knowledge_ids = example['knowledge_ids']
        # knowledge_mask = example['knowledge_mask']

        sample = {'idx': image_id,
                  'image': image,
                  'report_ids': report_ids,
                  'report_mask': report_masks,
                  'seq_length': seq_length,
                  'knowledge_ids': knowledge_ids,
                  # 'knowledge_mask': knowledge_mask
                  }
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        try:
            image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        except:
            example = self.examples[0]
            image_id = example['id']
            image_path = example['image_path']
            image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        knowledge_ids = example['knowledge_ids']
        # knowledge_mask = example['knowledge_mask']

        sample = {'idx': image_id,
                  'image': image,
                  'report_ids': report_ids,
                  'report_mask': report_masks,
                  'seq_length': seq_length,
                  'knowledge_ids': knowledge_ids,
                  # 'knowledge_mask': knowledge_mask
                  }
        return sample


class KnowledgeDataset(Dataset):
    """实体到知识做Embedding
        将知识编码成一句话，取avg_pooling位置的结果作为实体的Embedding
    """
    ORDER = ['suggestive_of', 'located_at', 'modify']
    def __init__(self, args, tokenizer):
        super(KnowledgeDataset, self).__init__()
        data = json.loads(open(args.knowledge_file).read())
        self.knowledge = data['knowledge']
        self.entity2id = data['entity2id']

        self.example = self.read_knowledge(self.entity2id)
        self.encodings = tokenizer(self.example, truncation=True, padding=True)

    def read_knowledge(self, entity2id):
        example = ['' for _ in range(len(entity2id))]

        for k, v in self.knowledge.items():
            entity = []
            for o in self.ORDER:
                entity.extend(v[o])
            example[entity2id[k]] = ', '.join(entity)
        return example

    def __getitem__(self, idx):
        # ecodings: {'input_ids': [], 'attention_mask': []}
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['idx'] = idx
        return item

    def __len__(self):
        return len(self.example)
