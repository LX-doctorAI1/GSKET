#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: knowledge_retrieval.py
"""
读取RadGraph数据，构建知识图谱库
根据生成的报告，从文本中提取关键实体，获取关键实体对应的知识
TODO: 分类别获取关键实体的知识：Anatomy获取所有知识，Observation获取多个实体3跳内有交集的部分知识
    观察到同一个实体两种类别同时存在，暂不考虑类别
遍历所有实体，检查报告中是否存在该实体，存在时，保留该实体
检查实体3跳内的知识，如果三跳内的目标实体与报告中存在的实体匹配，则将路径中间的实体作为知识
知识图谱需要的结构：
实体 - 实体类别
实体 - [关系，目标实体]
"""
import ipdb
import os
import json
import pandas as pd
import lmdb
import logging
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import io

import torch
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset

import six
import pyarrow as pa
from misc.utils import normalize

def read_facts(triple_file, cache=None):
    if not cache:
        cache = './data/cache'
    cache_file = os.path.join(cache, 'facts.pkl')
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            facts = pickle.load(f)
            return facts

    triple = pd.read_csv(triple_file, sep='\t', header=None)
    facts = {}

    for row in tqdm(triple.iterrows(), total=len(triple), ncols=100):
        subj = row[1][0].lower()
        rela = row[1][1].lower()
        obje = row[1][2].lower()

        if len(subj) <= 3 or len(obje) <= 3 or rela == 'modify':
            continue

        if subj not in facts:
            facts[subj] = []
        fact = [subj, rela, obje]
        if fact not in facts[subj]:
            facts[subj].append(fact)

    with open(cache_file, 'wb') as f:
        pickle.dump(facts, f)
    return facts


def find_facts(facts, report):
    retrieved_facts = []

    entities = []
    for k, v in facts.items():
        if k in report:
            entities.append(k)

    for entity in entities:
        # 第一跳
        first_facts = facts[entity]
        for f1 in first_facts:
            if f1[2] not in entities and f1[2] not in retrieved_facts:
                retrieved_facts.append(f1[2])
    return ' '.join(list(set(retrieved_facts)))


def batch_find_facts(facts, outputs, tokenizer, max_num=60):
    reports = tokenizer.decode_batch(outputs.cpu().detach().numpy())
    retrieved_facts = []
    for report in reports:
        retrieved_fact = find_facts(facts, report)
        report = tokenizer(retrieved_fact)
        report = [token for token in report if token > 2][:max_num]

        padd_facts = torch.zeros(max_num).long()
        padd_facts[:len(report)] = torch.Tensor(report).long()
        retrieved_facts.append(padd_facts)

    padd_retrieved_facts = torch.stack(retrieved_facts, 0)
    return padd_retrieved_facts

class retrieval_kg:
    def __init__(self, ner, tokenizer):
        entities_embeddings_filename = "/home/shuxinyang/code/KGMRG/data/RadGraph/RadGraph_MIMIC_KG_FromTrainSet_RotatE_entity.npy"
        entities_index_filename = "/home/shuxinyang/code/KGMRG/data/RadGraph/entities.tsv"

        self.entities_embeddings = np.load(entities_embeddings_filename)
        entities_index = pd.read_csv(entities_index_filename, sep='\t', header=None)

        self.entity_token2id = {}
        for row in entities_index.iterrows():
            self.entity_token2id[row[1][1]] = row[1][0]
        self.ner = ner
        self.tokenizer = tokenizer

    def retri_embeddings(self, report):
        entities = {'ANATOMY': [], 'OBSERVATION': [], 'OBSERVATION_MODIFIER': []}
        ner_docs = self.ner(report)
        for ent in ner_docs.ents:
            if ent.type in entities:
                entities[ent.type].append(ent.text)

        embeddings = []
        for name, tokens in entities.items():
            for token in tokens:
                if token in self.entity_token2id:
                    embeddings.append(self.entities_embeddings[self.entity_token2id[token]])

        return embeddings


    def batch_retri_embeddings(self, outputs, max_num=60):
        reports = self.tokenizer.decode_batch(outputs.cpu().detach().numpy())
        retrieved_emb = []
        for report in reports:
            retrieved_embeddings = self.retri_embeddings(report)
            emb_sz = 400

            padd_emb = torch.zeros((max_num, emb_sz))
            if len(retrieved_embeddings) > 0:
                padd_emb[:len(retrieved_embeddings)] = torch.Tensor(retrieved_embeddings)
            retrieved_emb.append(padd_emb)

        padd_retrieved_emb = torch.stack(retrieved_emb, 0)
        return padd_retrieved_emb

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--entity_file', type=str, default='../data/RadGraph/entities.tsv')
    parse.add_argument('--relation_file', type=str, default='../data/RadGraph/relations.tsv')
    parse.add_argument('--triple_file', type=str, default='../data/RadGraph/RadGraph_MIMIC_KG_FromTrainSet_train.tsv')
    parse.add_argument('--rel_feat_file', type=str, default='../data/RadGraph/RadGraph_MIMIC_KG_FromTrainSet_RotatE_relation.npy')
    parse.add_argument('--output', type=str, default='../data/RadGraph/RadGraph_MIMIC_relation_features.npy')

    args = parse.parse_args()

    # main(args)