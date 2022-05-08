#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/9
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: Triple2Knowledge.py

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
from stanza import Pipeline


def read_facts(knowledge_file, cache=None):
    # 从RadGraph的三元组知识库中读取知识
    # 返回：{头实体： 字符串形式的知识}
    if not cache:
        cache = '../data/cache'
    cache_file = os.path.join(cache, 'entity2knowledge_add_id.json')
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            facts = json.load(f)
            return facts

    triple = pd.read_csv(knowledge_file, sep='\t', header=None)
    facts = {}

    for row in tqdm(triple.iterrows(), total=len(triple), ncols=100):
        subj = row[1][0].lower()
        rela = row[1][1].lower()
        obje = row[1][2].lower()

        if len(subj) <= 3 or len(obje) <= 3:
            continue

        if subj not in facts:
            facts[subj] = {'located_at': set(), 'suggestive_of': set(), 'modify': set()}
        facts[subj][rela].add(obje)

    knowledge = {'entity2id': {}}
    knowledge['knowledge'] = facts

    for i, (entity, item) in enumerate(facts.items()):
        knowledge['entity2id'][entity] = i + 1
        for k, v in item.items():
            item[k] = list(v)

    with open(cache_file, 'w') as f:
        json.dump(knowledge, f)
    return facts


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--knowledge_file', type=str, default='../data/RadGraph/RadGraph_MIMIC_KG_FromTrainSet_train.tsv')
    parse.add_argument('--dataset', type=str, default='iu')
    parse.add_argument('--output', type=str, default='/home/shuxinyang/data/iu/dataset/entity2knowledge_add_id.json')

    args = parse.parse_args()

    read_facts(args.knowledge_file)
