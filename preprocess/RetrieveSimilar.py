import sys
sys.path.append('..')
import argparse
import os
import random
import os.path as osp
import json
import numpy as np
import pandas as pd
import scipy.spatial
from tqdm import tqdm
from misc import utils
import logging
from scipy.stats import entropy
from misc.utils import seed_everything


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def main(args, topk=10):
    with open(args.annpath, 'r') as f:
        ann = json.load(f)

    # 全部加载到内存，节省时间
    npy_file = {}
    files_id = {'train': [], 'val': [], 'test': [], 'retrieve': []}
    for split in ['train', 'val', 'test']:
        logging.info(f'precess {split} set')

        data = ann[split]
        for i, item in enumerate(tqdm(data)):
            dicom_id = item["id"]
            vec = np.load(osp.join(args.featurepath, dicom_id + '.npy'))
            vec = sigmoid(vec)
            npy_file[dicom_id] = vec
            files_id[split].append(dicom_id)

    retrieve_lists = files_id['train']
    logging.info(f"Number of Retrieve Set : {len(retrieve_lists)}")

    # 计算图片输出的label之间的KL散度，确定Conditional Case
    # 准备Pandas表格头
    outputs = {"filename": [], 'split': [], 'method': []}
    for i in range(topk):
        outputs["similar_{:d}".format(i)] = []
        outputs["score_{:d}".format(i)] = []

    # 处理三个split
    for split in ['train', 'val', 'test']:
        logging.info(f'Precess {split} set')
        file_lists = files_id[split]

        # 取出每个图像的预测向量，与Retrieve Set计算
        for target_id in tqdm(file_lists):
            conditions = []
            min_score = np.inf

            # MIMIC数据量太大，随机采样5000个
            sample_list = random.sample(retrieve_lists, min(5000, len(retrieve_lists)))
            for retrieve_id in sample_list:
                if target_id == retrieve_id:
                    continue

                vec_t, vec_p = npy_file[target_id], npy_file[retrieve_id]
                dist = entropy(vec_t, vec_p)
                conditions.append({"retrieve_id": retrieve_id, "score": dist})
                if dist < min_score:
                    min_score = dist
                # 如果计算量超过1000个，且最小的KL Div小于1e-4，停止计算
                if min_score < 0.0001 and len(conditions) > 1000:
                    break

            # 存储 top10
            conditions = sorted(conditions, key=lambda x: x["score"])
            outputs['filename'].append(target_id)
            outputs['split'].append(split)
            outputs['method'].append('KLDiv')
            for i, condition in enumerate(conditions[:topk]):
                outputs["similar_{:d}".format(i)].append(condition["retrieve_id"])
                outputs["score_{:d}".format(i)].append(condition["score"])

    csv_df = pd.DataFrame(outputs)
    csv_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    utils.set_logging()
    logging.info('====>> Retrieve Conditional Cases')
    seed_everything(0)
    logging.info('Set Random seed 0')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--annpath', type=str, default="/home/shuxinyang/data/iu/images/iu_2image/annotation.json")
    parser.add_argument('-f', '--featurepath', type=str, default="/home/shuxinyang/data/iu/feature/pretrain_on_mimic/")
    parser.add_argument('-o', '--output', type=str, default="/home/shuxinyang/data/iu/dataset/conditional_pair_kl.csv",
                        help="output file save dir")

    args = parser.parse_args()
    logging.info(str(args))
    main(args)

    # try:
    #     main(args)
    # except Exception as e:
    #     logging.error(e)
    # finally:
    #     logging.info('====>> Retrieve Conditional Cases Done')