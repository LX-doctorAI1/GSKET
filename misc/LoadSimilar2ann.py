#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/4
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: LoadSimilar2ann.py
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

def load_similar(args):
    pair = pd.read_csv(args.similar)

    pair_data = {}
    # 先添加similar的图片，最后添加要生成报告的图片
    for i, row in tqdm(pair.iterrows()):
        # if 'iu' in args.dataset:
        #     filename = 'CXR' + '-'.join(row['filename'].split('-')[:-1])
        # else:
        filename = row['filename']
        if filename not in pair_data:
            pair_data[filename] = []

        for j in range(10):
            similar_name = row['similar_{}'.format(j)]
            # if 'iu' in args.dataset:
            #     similar_name = 'CXR' + '-'.join(similar_name.split('-')[:-1])

            pair_data[filename].append(similar_name)

    return pair_data


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--ann', type=str, default='/home/shuxinyang/data/iu/dataset/annotation.json')
    parse.add_argument('--similar', type=str, default='/home/shuxinyang/data/iu/r2gen_split/pair_list.csv')
    parse.add_argument('--dataset', type=str, default='iu')
    parse.add_argument('--output', type=str, default='/home/shuxinyang/data/iu/dataset/ann_history_entity.json')

    args = parse.parse_args()

    with open(args.ann, 'r') as f:
        reports = json.load(f)
    ipdb.set_trace()
    similar_pair = load_similar(args)

    for split in reports.keys():
        for item in tqdm(reports[split]):
            idx = item['id']
            try:
                s = similar_pair[idx]
            except:
                print(idx)
                pass
            if len(s) > 10:
                s = s[:5] + s[10:15]
            item['history'] = s

    with open(args.output, 'w') as f:
        json.dump(reports, f)
