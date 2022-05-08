import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel


class BasicModel_v1(nn.Module):
    def __init__(self, args, tokenizer):
        super(BasicModel_v1, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        # 图像编码器
        self.visual_extractor = VisualExtractor(args)
        if 'iu' in args.dataset_name:
            self.visual_encoder = self.forward_iu_xray
        else:
            self.visual_encoder = self.forward_mimic_cxr

        self.proj_v1 = nn.Linear(args.d_vf, args.d_model)
        self.proj_v2 = nn.Linear(args.d_vf, args.d_model)

        self.encoder_decoder = TransformerModel(args, tokenizer)
        self.proj = nn.Linear(args.num_labels, args.d_vf)

        # self.init_weight(self.proj_v1)
        # self.init_weight(self.proj_v2)

    @staticmethod
    def init_weight(f):
        nn.init.kaiming_normal_(f.weight)
        f.bias.data.fill_(0)

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

    def forward(self, data, mode='train'):
        images = data['images']
        targets = data['targets']

        if self.args.dataset_name == 'iu_xray':
            att_feats, fc_feats, out_labels = self.forward_iu_xray(images)
        else:
            att_feats, fc_feats, out_labels = self.forward_mimic_cxr(images)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, opt=self.args, mode='sample')
        else:
            raise ValueError
        return output
