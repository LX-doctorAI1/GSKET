#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/5
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: KGModel.py

import ipdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stanza import Pipeline

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel, clones
from modules.text_encoder import TextEncoder, MHA_FF
from modules.knowledge_retrieval import retrieval_kg

from misc.KnowledgeEmbedding import PretrainedEmbeddings


class KGMultiHeadAttention(nn.Module):
    """ 用于知识图谱的定制Attention
        知识图谱包含不同类型的关系，
        不只是在节点上进行在attention，
        而且将边之间的关系作为bias引入 attetion
    """

    def __init__(self, h, d_model, dropout=0.1, return_score=False):
        "Take in model size and number of heads."
        super(KGMultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.return_score = return_score

    def forward(self, query, key, value, mask=None, bias=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask,
                                      dropout=self.dropout, bias=bias)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None, bias=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        # [bs, h, q_len, v_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # Add Bias for incorporating graph structure information
        if bias is not None:
            scores += bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class KG_MHA_FF(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        from modules.Transformer import SublayerConnection, PositionwiseFeedForward

        super(KG_MHA_FF, self).__init__()
        self.self_attn = KGMultiHeadAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, feats, bias=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, feats, feats, bias=bias))
        return self.sublayer[1](x, self.feed_forward)


class KGModel_v1(nn.Module):
    """ 融合知识的两级生成模型（V1）
        - 第一级图像和基于图像使用权威知识图谱生成粗略报告
        - 然后使用NER模型从报告中提取实体，根据实体获取医学知识图谱对应知识的embedding
        - 将知识融合，再生成最终的报告
        - 训练过程先预训练初级transformer 8轮，然后同时训练两级
    """
    KG_ENTITY_DIM = 400
    NER_BATCH_SIZE = 256
    MAX_KNOWS = 60  # 知识的最大数量

    def __init__(self, args, tokenizer):
        super(KGModel_v1, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.tgt_vocab = len(tokenizer.idx2token) + 1

        # 知识实体的embedding
        embeddings = PretrainedEmbeddings.load_embeddings(args.pretrained_embedding)
        self.know_ent_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False,
                                                    padding_idx=PretrainedEmbeddings.INDEX_PAD)
        self.proj_bert_embedding = nn.Linear(768, args.d_model)

        # 图像编码器
        self.visual_extractor = VisualExtractor(args)
        if 'iu' in args.dataset_name:
            self.visual_encoder = self.forward_iu_xray
        else:
            self.visual_encoder = self.forward_mimic_cxr
        self.proj_v1 = nn.Linear(args.d_vf, args.d_model)
        self.proj_v2 = nn.Linear(args.d_vf, args.d_model)

        # image + knowledge -> text
        self.generator = TransformerModel(args, tokenizer)

        # 图像和知识图谱的Attention
        self.attention = KG_MHA_FF(d_model=args.d_model,
                                   d_ff=args.d_ff,
                                   h=args.num_heads,
                                   dropout=args.dropout)
        # 图像和医学知识的Attention
        self.know_att = MHA_FF(d_model=args.d_model,
                               d_ff=args.d_ff,
                               h=args.num_heads,
                               dropout=args.dropout)

        # 加载医生标注的知识图谱的节点和边，用于bias
        self.chest_embedding, self.relation_embedding = self.load_chest_KG()
        self.proj_entity = nn.Linear(self.KG_ENTITY_DIM, args.d_model)
        self.proj_relation = nn.Linear(self.KG_ENTITY_DIM, 1)

        self.init_weight(self.proj_v1)
        self.init_weight(self.proj_v2)
        self.init_weight(self.proj_entity)
        self.init_weight(self.proj_relation)
        self.init_weight(self.proj_bert_embedding)

    @staticmethod
    def init_weight(f):
        nn.init.kaiming_normal_(f.weight)
        f.bias.data.fill_(0)

    @classmethod
    def load_ner(cls):
        config = {'tokenize_batch_size': cls.NER_BATCH_SIZE, 'ner_batch_size': cls.NER_BATCH_SIZE}
        return Pipeline(lang='en', package='radiology', processors={'tokenize': 'default', 'ner': 'radiology'},
                        **config)

    def load_chest_KG(self):
        chest_entity = np.load(self.args.entity_file)
        chest_embedding = torch.FloatTensor(chest_entity)

        relation_embedding = np.load(self.args.relation_file)
        relation_embedding = torch.FloatTensor(relation_embedding)

        return chest_embedding, relation_embedding

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images):
        att_feats_0, fc_feats_0, out_labels = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, out_labels = self.visual_extractor(images[:, 1])

        fc_feats = torch.stack([fc_feats_0, fc_feats_1], dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        fc_feats = self.proj_v1(fc_feats)
        att_feats = self.proj_v2(att_feats)

        return att_feats, fc_feats, out_labels

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        fc_feats = self.proj_v1(fc_feats.unsqueeze(1))
        att_feats = self.proj_v2(att_feats)
        return att_feats, fc_feats, out_labels


    def forward(self, data, mode='train', epoch=0):
        images = data['images']
        targets = data['targets']
        knowledges = data['knowledges']
        chest_embedding = self.chest_embedding.to(images.device)
        relation_embedding = self.relation_embedding.to(images.device)

        # 提取视觉信息
        att_feats, fc_feats, label_feats = self.visual_encoder(images)

        bsz, _, d_model = att_feats.shape
        # 视觉特征在医生手工标注的的知识图谱上attention
        entity_embedding = self.proj_entity(chest_embedding).unsqueeze(0).expand(bsz, -1, -1)
        bias = self.proj_relation(relation_embedding).squeeze()
        clinic_embed = self.attention(fc_feats, entity_embedding, bias=bias)

        knowledges = self.know_ent_embeddings(knowledges)
        knowledges = self.proj_bert_embedding(knowledges)
        know_embed = self.know_att(att_feats, knowledges)

        # 合并用于最后的生成
        att_feats = torch.cat([att_feats, clinic_embed, know_embed], dim=1)
        if mode == 'train':
            output = self.generator(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.generator(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        return output


class KGModel_vWOExternalKG(KGModel_v1):
    def forward(self, data, mode='train', epoch=0):
        images = data['images']
        targets = data['targets']
        knowledges = data['knowledges']

        # 提取视觉信息
        att_feats, fc_feats, label_feats = self.visual_encoder(images)

        knowledges = self.know_ent_embeddings(knowledges)
        knowledges = self.proj_bert_embedding(knowledges)
        know_embed = self.know_att(att_feats, knowledges)

        # 合并用于最后的生成
        att_feats = torch.cat([att_feats, know_embed], dim=1)
        if mode == 'train':
            output = self.generator(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.generator(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        return output


class KGModel_vWOInternalKG(KGModel_v1):
    def forward(self, data, mode='train', epoch=0):
        images = data['images']
        targets = data['targets']
        knowledges = data['knowledges']
        chest_embedding = self.chest_embedding.to(images.device)
        relation_embedding = self.relation_embedding.to(images.device)

        # 提取视觉信息
        att_feats, fc_feats, label_feats = self.visual_encoder(images)

        bsz, _, d_model = att_feats.shape
        # 视觉特征在医生手工标注的的知识图谱上attention
        entity_embedding = self.proj_entity(chest_embedding).unsqueeze(0).expand(bsz, -1, -1)
        bias = self.proj_relation(relation_embedding).squeeze()
        clinic_embed = self.attention(fc_feats, entity_embedding, bias=bias)

        # 合并用于最后的生成
        att_feats = torch.cat([att_feats, clinic_embed], dim=1)
        if mode == 'train':
            output = self.generator(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.generator(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        return output


class KGModel_vWMHA(KGModel_v1):
    """对比实验
    把GMHA换成普通的MHA
    """
    def __init__(self, args, tokenizer):
        super(KGModel_vWMHA, self).__init__(args, tokenizer)
        # 图像和知识图谱的Attention
        self.attention = KG_MHA_FF(d_model=args.d_model,
                                   d_ff=args.d_ff,
                                   h=args.num_heads,
                                   dropout=args.dropout)

    def forward(self, data, mode='train', epoch=0):
        images = data['images']
        targets = data['targets']
        knowledges = data['knowledges']
        chest_embedding = self.chest_embedding.to(images.device)
        relation_embedding = self.relation_embedding.to(images.device)

        # 提取视觉信息
        att_feats, fc_feats, label_feats = self.visual_encoder(images)

        bsz, _, d_model = att_feats.shape
        # 视觉特征在医生手工标注的的知识图谱上attention
        entity_embedding = self.proj_entity(chest_embedding).unsqueeze(0).expand(bsz, -1, -1)
        bias = self.proj_relation(relation_embedding).squeeze()
        clinic_embed = self.attention(fc_feats, entity_embedding, bias=bias)

        knowledges = self.know_ent_embeddings(knowledges)
        knowledges = self.proj_bert_embedding(knowledges)
        know_embed = self.know_att(att_feats, knowledges)

        # 合并用于最后的生成
        att_feats = torch.cat([att_feats, clinic_embed, know_embed], dim=1)
        if mode == 'train':
            output = self.generator(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.generator(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        return output