#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/6
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: KnowledgeEmbedding.py
import os
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


def get_bert_dataloader(args, tokenizer):
    dataset = KnowledgeDataset(args, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
    return dataset, dataloader

class PretrainedEmbeddings:
    PAD = '<pad>'
    INDEX_PAD = 0

    @classmethod
    def load_embeddings(cls, path):
        """
        Load pre-trained word embedding.
        :param path: A path to pre-trained word embeddings
        :return: A tensor of pre-trained word embeddings
        """
        word_idxs = {'__PAD__': cls.INDEX_PAD}

        pretrained_embeddings = np.load(path)
        num, dim = pretrained_embeddings.shape

        embeddings = np.zeros((num + len(word_idxs), dim), dtype='float32')

        embeddings[cls.INDEX_PAD] = np.random.uniform(low=-1, high=1, size=dim)

        embeddings[cls.INDEX_PAD + 1:] = pretrained_embeddings
        embeddings = torch.tensor(embeddings)

        return embeddings


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
        self.encodings = tokenizer(self.example, truncation=True, padding=True, max_length=512)

    def read_knowledge(self, entity2id):
        example = ['' for _ in range(len(entity2id))]
        count = 0
        for k, v in self.knowledge.items():
            entity = []
            for o in self.ORDER:
                entity.extend(v[o])
            if len(entity) > 510:
                print(k)
                count += 1
            example[entity2id[k]] = ' '.join(entity[:128])
        print(count)
        return example

    def __getitem__(self, idx):
        # ecodings: {'input_ids': [], 'attention_mask': []}
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['idx'] = idx
        return item

    def __len__(self):
        return len(self.example)


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if hasattr(args, 'bert_model'):
        model_name_or_path = args.bert_model
    else:
        model_name_or_path = 'bert-base-uncased'
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    bert_dataset, bert_dataloader = get_bert_dataloader(args, tokenizer=bert_tokenizer)

    model = AutoModel.from_pretrained(model_name_or_path).to(device)

    bert_embedding = np.zeros((len(bert_dataset), 768))
    for batch in tqdm(bert_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        idx = batch['idx']
        outputs = model(input_ids, attention_mask=attention_mask)
        embedding = outputs[1].detach().cpu().numpy()

        bert_embedding[idx] = embedding

    np.save(args.output, bert_embedding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge_file', type=str, default='/home/shuxinyang/data/iu/dataset/entity2knowledge_add_id.json')
    parser.add_argument('--output', type=str, default='/home/shuxinyang/data/iu/dataset/knowledge_embedding')
    parser.add_argument('--bert_model', type=str, default="Bio_ClinicalBERT")
    parser.add_argument('--batch_size', type=int, default=16)

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    args = parser.parse_args()
    main(args)