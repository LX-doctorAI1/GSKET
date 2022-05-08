# -*- coding: utf-8 -*-
# @Time    : 2021/5/31
# @Author  : Aspen Stars
# @Contact : aspenstars@qq.com
# @FileName: GConv.py
import ipdb
import math
import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


"""copy from https://docs.dgl.ai/guide_cn/training-node.html#guide-cn-training-node-classification"""
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean', activation=nn.ReLU())
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean', activation=nn.ReLU())
        self.conv3 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean', activation=nn.ReLU())

    def forward(self, graph, inputs):
        out = torch.zeros_like(inputs)
        # 输入是节点的特征
        for i in range(inputs.size(0)):
            h = self.conv1(graph, inputs[i])
            h = F.relu(h)
            h = self.conv2(graph, h)
            h = F.relu(h)
            h = self.conv3(graph, h)
            out[i] = h
        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_node=126, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Sequential(nn.Linear(in_features, out_features),
                                    nn.BatchNorm1d(num_node),
                                    )
        nn.init.kaiming_normal_(self.linear[0].weight)
        self.linear[0].bias.data.fill_(0)

    def forward(self, input, adj):
        support = self.linear(input)
        output = torch.matmul(adj.to_dense(), support)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nnode, dropout=0.1):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nfeat, nnode)
        self.gc2 = GraphConvolution(nfeat, nfeat, nnode)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x + F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x + self.gc2(x, adj)
        return x
