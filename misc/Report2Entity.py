#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/4
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: Report2Entity.py

"""
预处理步骤，加快训练
使用Stanza从报告中提取实体
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
from stanza import Pipeline


class NER:
    NER_BATCH_SIZE = 256

    def __init__(self, args):
        self.args = args

        config = {'tokenize_batch_size': self.NER_BATCH_SIZE, 'ner_batch_size': self.NER_BATCH_SIZE}
        self.ner = Pipeline(lang='en', package='radiology', processors={'tokenize': 'default', 'ner': 'radiology'},
                            **config)
        with open(args.ann, 'r') as f:
            self.reports = json.load(f)
        # self.facts = self.read_facts(args.knowledge_file)

    def extract(self, dataset='iu'):

        for split in self.reports.keys():
            for item in tqdm(self.reports[split]):
                report = item['report'].lower()

                docs = self.ner(report)
                # 将同一类型的实体放在一起，避免语义混乱
                entities = {'ANATOMY': [], 'OBSERVATION': [], 'OBSERVATION_MODIFIER': [], 'ANATOMY_MODIFIER': []}

                for ent in docs.ents:
                    if ent.type in entities:
                        entities[ent.type].append(ent.text)

                item['entity'] = entities

        with open(self.args.output, 'w') as f:
            json.dump(self.reports, f)

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
    for i, (entity, item) in enumerate(facts.items()):
        knowledge['entity2id'][entity] = i + 1
        for k, v in item.items():
            item[k] = list(v)

    knowledge['knowledge'] = facts

    with open(cache_file, 'w') as f:
        json.dump(knowledge, f)
    return facts


def helper(knowledge_file, cache=None):
    # 从RadGraph的三元组知识库中读取知识
    # 返回：{头实体： 字符串形式的知识}
    if not cache:
        cache = '../data/cache'
    cache_file = os.path.join(cache, 'entity2knowledge_only_subject.json')
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            facts = json.load(f)

    knowledge = {'entity2id': {}}
    for i, (entity, item) in enumerate(facts.items()):
        knowledge['entity2id'][entity] = i + 1

    knowledge['knowledge'] = facts

    cache_file = os.path.join(cache, 'entity2knowledge_add_id.json')
    with open(cache_file, 'w') as f:
        json.dump(knowledge, f)
    return facts


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--ann', type=str, default='/home/shuxinyang/data/iu/dataset/annotation.json')
    parse.add_argument('--knowledge_file', type=str, default='../data/RadGraph/RadGraph_MIMIC_KG_FromTrainSet_train.tsv')
    parse.add_argument('--dataset', type=str, default='iu')
    parse.add_argument('--output', type=str, default='/home/shuxinyang/data/iu/dataset/ann_entity.json')

    args = parse.parse_args()

    report2entity = NER(args)
    report2entity.extract(args.dataset)
    # main(args)
    # read_facts(args.knowledge_file)
    # helper(args.knowledge_file)