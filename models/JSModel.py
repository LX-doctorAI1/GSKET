#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/30
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: JSModel.py

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


class JSModel_v1(nn.Module):
    """ 融合知识的两级生成模型（V1）
        - 第一级图像和基于图像使用权威知识图谱生成粗略报告
        - 然后使用NER模型从报告中提取实体，根据实体获取医学知识图谱对应知识的embedding
        - 将知识融合，再生成最终的报告
        - 训练过程先预训练初级transformer 8轮，然后同时训练两级
    """
    KG_ENTITY_DIM = 400
    NER_BATCH_SIZE = 256

    def __init__(self, args, tokenizer):
        super(JSModel_v1, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        # 图像编码器
        self.visual_extractor = VisualExtractor(args)
        self.visual_encoder = self.forward_mimic_cxr
        self.proj_v1 = nn.Linear(args.d_vf, args.d_model)
        self.proj_v2 = nn.Linear(args.d_vf, args.d_model)

        # 初级和高级报告生成，初级报告再编码
        self.junior = TransformerModel(args, tokenizer)
        self.senior = TransformerModel(args, tokenizer)
        self.draft_encoder = TextEncoder(d_model=args.d_model,
                                         d_ff=args.d_ff,
                                         num_layers=args.num_layers,
                                         tgt_vocab=self.tgt_vocab,
                                         num_labels=args.num_labels,
                                         h=args.num_heads,
                                         dropout=args.dropout,
                                         src_embed=self.junior.model.tgt_embed)
        # 使用相同的embedding
        self.senior.model.tgt_embed = self.junior.model.tgt_embed

        # 图像和知识图谱的Attention
        self.attention = MHA_FF(d_model=args.d_model,
                                d_ff=args.d_ff,
                                h=args.num_heads,
                                dropout=args.dropout)
        self.proj_clinic_kg = nn.Linear(400, args.d_model)

        # 加载从初级报告中提取实体的NER模型
        self.ner = self.load_ner()
        self.retrieval = retrieval_kg(self.ner, self.tokenizer)
        self.proj_entity = nn.Linear(self.KG_ENTITY_DIM, args.d_model)

        self.init_weight(self.proj_v1)
        self.init_weight(self.proj_v2)
        self.init_weight(self.proj_entity)
        self.init_weight(self.proj_clinic_kg)

    @staticmethod
    def init_weight(f):
        nn.init.kaiming_normal_(f.weight)
        f.bias.data.fill_(0)

    @classmethod
    def load_ner(cls):
        config = {'tokenize_batch_size': cls.NER_BATCH_SIZE, 'ner_batch_size': cls.NER_BATCH_SIZE}
        return Pipeline(lang='en', package='radiology', processors={'tokenize': 'default', 'ner': 'radiology'},
                        **config)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats, out_labels = self.visual_extractor(images)
        fc_feats = self.proj_v1(fc_feats)
        att_feats = self.proj_v2(att_feats)
        return att_feats, fc_feats, out_labels

    def load_chest_KG(self):
        chest_entity = np.load(self.args.entity_file)
        chest_embedding = torch.FloatTensor(chest_entity)
        return chest_embedding


    def forward(self, images, entity_embedding, con_report, targets=None, labels=None, facts=None, mode='train', epoch=0):
        # 提取视觉信息
        att_feats, fc_feats, label_feats = self.visual_encoder(images[:, -1])

        if mode == 'train':
            junior_output = self.junior(fc_feats, att_feats, targets, mode='forward')
            draft = targets[:, 1:]
        elif mode == 'sample':
            junior_output, _ = self.junior(fc_feats, att_feats, opt=self.args, mode='sample')
            draft = junior_output
        else:
            raise ValueError
        # ipdb.set_trace()
        # 前8个epoch只训练初级Transformer
        if epoch < 8:
            return junior_output, None

        # 第二级不再更新前面的模型
        fc_feats = fc_feats.detach()
        # att_feats = att_feats.detach()

        bsz, _, d_model = att_feats.shape
        # 视觉特征在医生手工标注的的知识图谱上attention
        fc_feats = fc_feats.unsqueeze(1)
        entity_embedding = self.proj_clinic_kg(entity_embedding).unsqueeze(0)
        entity_embedding = entity_embedding.expand(bsz, -1, -1)
        clinic_embed = self.attention(fc_feats, entity_embedding)

        # 将之前生成的报告再编码
        draft_embed, draft_embed_avg, draft_labels = self.draft_encoder(draft)

        # 提取知识的embedding
        entity_embed = self.retrieval.batch_retri_embeddings(draft)
        entity_embed = self.proj_entity(entity_embed.to(images.device))

        # 合并用于最后的生成
        att_feats = torch.cat([att_feats, draft_embed, entity_embed, clinic_embed], dim=1)
        if mode == 'train':
            senior_output = self.senior(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            senior_output, _ = self.senior(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        return junior_output, senior_output


class JSModel_v12(JSModel_v1):
    """ 融合知识的两级生成模型（V1） - 将MultiHead attention模块换成图多头注意力模块
        - 第一级图像和基于图像使用权威知识图谱生成粗略报告
        - 然后使用NER模型从报告中提取实体，根据实体获取医学知识图谱对应知识的embedding
        - 将知识融合，再生成最终的报告
        - 训练过程先预训练初级transformer 8轮，然后同时训练两级
    """
    KG_ENTITY_DIM = 400
    NER_BATCH_SIZE = 256

    def __init__(self, args, tokenizer):
        super(JSModel_v12, self).__init__(args, tokenizer)
        # 区别在于图像和权威知识图谱的Attention换成了KG_MHA_FF，增加了边的特征作为bias

        # 加载医生标注的知识图谱的节点和边，用于bias
        self.chest_embedding, self.relation_embedding = self.load_chest_KG()
        self.proj_relation = nn.Linear(400, 1)

        # 图像和知识图谱的Attention
        self.attention = KG_MHA_FF(d_model=args.d_model,
                                   d_ff=args.d_ff,
                                   h=args.num_heads,
                                   dropout=args.dropout)

        self.init_weight(self.proj_relation)

    def load_chest_KG(self):
        chest_entity = np.load(self.args.entity_file)
        chest_embedding = torch.FloatTensor(chest_entity)

        relation_embedding = np.load(self.args.relation_file)
        relation_embedding = torch.FloatTensor(relation_embedding)

        return chest_embedding, relation_embedding


    def forward(self, images, entity_embedding, con_report, targets=None, labels=None, facts=None, mode='train', epoch=0):
        # 提取视觉信息
        att_feats, fc_feats, label_feats = self.visual_encoder(images[:, -1])

        if mode == 'train':
            junior_output = self.junior(fc_feats, att_feats, targets, mode='forward')
            draft = targets[:, 1:]
        elif mode == 'sample':
            junior_output, _ = self.junior(fc_feats, att_feats, opt=self.args, mode='sample')
            draft = junior_output
        else:
            raise ValueError
        # ipdb.set_trace()
        # 前8个epoch只训练初级Transformer
        if epoch < 8:
            return junior_output, None

        # 第二级不再更新前面的模型
        fc_feats = fc_feats.detach()
        # att_feats = att_feats.detach()

        bsz, _, d_model = att_feats.shape
        # 视觉特征在医生手工标注的的知识图谱上attention
        fc_feats = fc_feats.unsqueeze(1)
        entity_embedding = self.proj_clinic_kg(entity_embedding).unsqueeze(0)
        entity_embedding = entity_embedding.expand(bsz, -1, -1)
        bias = self.proj_relation(self.relation_embedding.to(images.device)).squeeze()

        clinic_embed = self.attention(fc_feats, entity_embedding, bias=bias)

        # 将之前生成的报告再编码
        draft_embed, draft_embed_avg, draft_labels = self.draft_encoder(draft)

        # 提取知识的embedding
        entity_embed = self.retrieval.batch_retri_embeddings(draft)
        entity_embed = self.proj_entity(entity_embed.to(images.device))

        # 合并用于最后的生成
        att_feats = torch.cat([att_feats, draft_embed, entity_embed, clinic_embed], dim=1)
        if mode == 'train':
            senior_output = self.senior(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            senior_output, _ = self.senior(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        return junior_output, senior_output


class JSModel_v2(JSModel_v1):
    """在生成草稿报告的时候使用医学知识图谱，生成最终报告的时候使用权威知识图谱"""
    def forward(self, images, entity_embedding, con_report, targets=None, labels=None, facts=None, mode='train', epoch=0):
        # 提取视觉信息
        att_feats, fc_feats, label_feats = self.visual_encoder(images[:, -1])

        bsz, _, d_model = att_feats.shape
        # 视觉特征在医生手工标注的的知识图谱上attention
        fc_feats = fc_feats.unsqueeze(1)
        entity_embedding = self.proj_clinic_kg(entity_embedding).unsqueeze(0)
        entity_embedding = entity_embedding.expand(bsz, -1, -1)
        clinic_embed = self.attention(fc_feats, entity_embedding)

        junior_feats = torch.cat([att_feats, clinic_embed], dim=1)
        if mode == 'train':
            junior_output = self.junior(fc_feats, junior_feats, targets, mode='forward')
            draft = targets[:, 1:]
        elif mode == 'sample':
            junior_output, _ = self.junior(fc_feats, junior_feats, opt=self.args, mode='sample')
            draft = junior_output
        else:
            raise ValueError
        # ipdb.set_trace()
        # 前8个epoch只训练初级Transformer
        if epoch < 8:
            return junior_output, None

        # 第一级负责更新CNN，第二级生成时不再更新CNN
        att_feats = att_feats.detach()
        # 将之前生成的报告再编码，用于辅助之后的生成
        draft_embed, draft_embed_avg, draft_labels = self.draft_encoder(draft)

        # 提取知识的embedding
        entity_embed = self.retrieval.batch_retri_embeddings(draft)
        entity_embed = self.proj_entity(entity_embed.to(images.device))

        # 合并用于最后的生成
        att_feats = torch.cat([att_feats, entity_embed, draft_embed], dim=1)
        if mode == 'train':
            senior_output = self.senior(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            senior_output, _ = self.senior(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        return junior_output, senior_output


class JSModel_v22(JSModel_v12):
    """在生成草稿报告的时候使用医学知识图谱，生成最终报告的时候使用权威知识图谱"""

    def forward(self, images, entity_embedding, con_report, targets=None, labels=None, facts=None, mode='train',
                epoch=0):
        # 提取视觉信息
        att_feats, fc_feats, label_feats = self.visual_encoder(images[:, -1])

        bsz, _, d_model = att_feats.shape
        # 视觉特征在医生手工标注的的知识图谱上attention
        fc_feats = fc_feats.unsqueeze(1)
        entity_embedding = self.proj_clinic_kg(entity_embedding).unsqueeze(0)
        entity_embedding = entity_embedding.expand(bsz, -1, -1)
        bias = self.proj_relation(self.relation_embedding.to(images.device)).squeeze()

        clinic_embed = self.attention(fc_feats, entity_embedding, bias=bias)

        junior_feats = torch.cat([att_feats, clinic_embed], dim=1)
        if mode == 'train':
            junior_output = self.junior(fc_feats, junior_feats, targets, mode='forward')
            draft = targets[:, 1:]
        elif mode == 'sample':
            junior_output, _ = self.junior(fc_feats, junior_feats, opt=self.args, mode='sample')
            draft = junior_output
        else:
            raise ValueError
        # ipdb.set_trace()
        # 前8个epoch只训练初级Transformer
        if epoch < 8:
            return junior_output, None

        # 第一级负责更新CNN，第二级生成时不再更新CNN
        att_feats = att_feats.detach()
        # 将之前生成的报告再编码，用于辅助之后的生成
        draft_embed, draft_embed_avg, draft_labels = self.draft_encoder(draft)

        # 提取知识的embedding
        entity_embed = self.retrieval.batch_retri_embeddings(draft)
        entity_embed = self.proj_entity(entity_embed.to(images.device))

        # 合并用于最后的生成
        att_feats = torch.cat([att_feats, entity_embed, draft_embed], dim=1)
        if mode == 'train':
            senior_output = self.senior(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            senior_output, _ = self.senior(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError

        return junior_output, senior_output