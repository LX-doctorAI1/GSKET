#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/5
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: main_knowledge.py
import warnings
warnings.simplefilter("ignore", UserWarning)
import ipdb

import logging
import numpy as np
import json
from scipy import sparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
import models

from config import opts
from misc import utils


class Trainer_Graph(Trainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # 加载医生手工标注的知识图谱
        self.entity_embedding, self.relation_embedding = self.load_knowledge_graph_embedding()
        self.entity_embedding = self.entity_embedding.to(self.device)
        self.relation_embedding = self.relation_embedding.to(self.device)

    def load_knowledge_graph_embedding(self):
        entity_embedding_file = self.args.entity_file
        entity_embedding = np.load(entity_embedding_file)
        entity_embedding = torch.FloatTensor(entity_embedding)

        relation_embedding = np.load(self.args.relation_file)
        relation_embedding = torch.FloatTensor(relation_embedding)

        return entity_embedding, relation_embedding

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()

        t = tqdm(self.train_dataloader, ncols=80)
        for batch_idx, batch in enumerate(t):

            images = batch['images'].to(self.device)
            reports_ids = batch['targets'].to(self.device)
            knowledges = batch['knowledges'].to(self.device)
            reports_mask = batch['reports_mask'].to(self.device)

            data = {'images': images,
                    'targets': reports_ids,
                    'knowledges': knowledges}
            self.optimizer.zero_grad()
            output = self.model(data, mode='train', epoch=epoch)

            loss = self.criterion(output, reports_ids, reports_mask)

            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            t.set_description(f'loss:{loss.item():.3}')

            if self.args.test_steps > 0 and epoch > 1 and (batch_idx + 1) % self.args.test_steps == 0:
                self.test_step(epoch, batch_idx + 1)
                self.model.train()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        ilog = self._test_step(epoch, 0, 'val')
        log.update(**ilog)

        ilog = self._test_step(epoch, 0, 'test')
        log.update(**ilog)

        self.lr_scheduler.step()

        return log

    def _test_step(self, epoch, iters=0, mode='test'):
        log = {}
        self.model.eval()

        data_loader = self.val_dataloader if mode == 'val' else self.test_dataloader
        with torch.no_grad():
            # if epoch < 8:
            #     val_gts, val_res, val_idxs = ["a b c d"], ["d e f g"], ["1"]
            # else:
            val_gts, val_res, val_idxs = [], [], []
            t = tqdm(data_loader, ncols=80)
            for batch_idx, batch in enumerate(t):
                images_id = batch['images_id']
                images = batch['images'].to(self.device)
                reports_ids = batch['targets'].to(self.device)
                knowledges = batch['knowledges'].to(self.device)

                data = {'images': images,
                        'targets': reports_ids,
                        'knowledges': knowledges}
                output = self.model(data, mode='sample', epoch=epoch)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                val_idxs.extend(images_id)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{f'{mode}_' + k: v for k, v in val_met.items()})
            self._output_generation(val_res, val_gts, val_idxs, epoch, iters, mode)
        return log


def main():
    # parse arguments
    # args = parse_agrs()
    args = opts.parse_opt('KNOW')
    logging.info(str(args))

    # fix random seeds
    utils.seed_everything(args.seed)
    logging.info(f'Set random seed : {args.seed}')

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model_name = f"KGModel_v{args.version}"
    logging.info(f"Model name: {model_name} \tModel Layers:{args.num_layers}")
    model = getattr(models, model_name)(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer_Graph(model, criterion, metrics, optimizer, args, lr_scheduler,
                            train_dataloader, val_dataloader, test_dataloader)
    trainer.train()
    logging.info(str(args))


if __name__ == '__main__':
    main()
