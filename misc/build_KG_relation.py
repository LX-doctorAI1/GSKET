#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/19
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: build_KG_relation.py
import ipdb
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def main(args):
    entity = pd.read_csv(args.entity_file, sep='\t', header=None)
    relation = pd.read_csv(args.relation_file, sep='\t', header=None)
    triple = pd.read_csv(args.triple_file, sep='\t', header=None)
    r = np.load(args.rel_feat_file)

    entity_token2id = {}
    relation_token2id = {}

    for row in relation.iterrows():
        relation_token2id[row[1][1]] = row[1][0]

    for row in entity.iterrows():
        entity_token2id[row[1][1]] = row[1][0]

    rel_feats = np.zeros((len(entity_token2id), len(entity_token2id), 400))
    for row in tqdm(triple.iterrows()):
        subj = entity_token2id[row[1][0]]
        rela = relation_token2id[row[1][1]]
        obje = entity_token2id[row[1][2]]
        rel_feats[subj, obje] = r[rela]

    ipdb.set_trace()
    m_rel_feats = np.mean(rel_feats, axis=1)
    np.save(args.output, m_rel_feats)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--entity_file', type=str, default='../data/RadGraph/entities.tsv')
    parse.add_argument('--relation_file', type=str, default='../data/RadGraph/relations.tsv')
    parse.add_argument('--triple_file', type=str, default='../data/RadGraph/RadGraph_MIMIC_KG_FromTrainSet_train.tsv')
    parse.add_argument('--rel_feat_file', type=str, default='../data/RadGraph/RadGraph_MIMIC_KG_FromTrainSet_RotatE_relation.npy')
    parse.add_argument('--output', type=str, default='../data/RadGraph/RadGraph_MIMIC_relation_features.npy')

    args = parse.parse_args()

    main(args)