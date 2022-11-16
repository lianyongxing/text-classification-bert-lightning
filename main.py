#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : xhs_trainer.py
@author: xiaoya li
@contact : xiaoya_li@shannonai.com
@date  : 2020/11/20 11:02
@version: 1.0
@desc  :
"""

import os
import json
import logging
import argparse
import warnings

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# from datasets.tnews_dataset import TNewsDataset
# from datasets.collate_functions import collate_to_max_length
# from models.modeling_glycebert import GlyceBertForSequenceClassification
from datasets.xhs_datasets import XHSDataset
from models.bert import Bert

class BertTextClassificationTask(pl.LightningModule):

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        self.model = Bert(self.bert_dir)
        self.criterion = CrossEntropyLoss()

        self.num_labels = 2
            # len(TNewsDataset.get_labels())
        # self.bert_config = BertConfig.from_pretrained(self.bert_dir,
        #                                               output_hidden_states=False,
        #                                               num_labels=self.num_labels)
        # self.model = GlyceBertForSequenceClassification.from_pretrained(self.bert_dir,
        #                                                                 config=self.bert_config)


        self.acc = pl.metrics.Accuracy(num_classes=self.num_labels)

        # self.num_gpus = 1
        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "eval_result_log.txt"),
                            level=logging.INFO)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        optimizer = AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)  # AdamW优化器
        # num_gpus = self.num_gpus
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(self.train_dataloader()),
                                                    num_training_steps=self.args.max_epochs * len(self.train_dataloader()))

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, input_types, input_masks):
        return self.model(input_ids=input_ids, token_type_ids=input_types, attention_mask=input_masks)

    def compute_loss_and_acc(self, batch):
        ids, att, tpe, lab = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label']
        y = lab.long()
        y_hat = self.forward(ids, tpe, att)

        # compute loss
        loss = self.criterion(y_hat, y)
        # compute acc
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        acc = self.acc(predict_labels, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    # def val_dataloader(self):
    #     return self.get_dataloader("dev")

    def get_encodings(self, path, mode='train'):
        datas = pd.read_csv(path)[:1000]
        texts = datas['content_filter'].tolist()
        if mode == 'train':
            labs = datas['lab'].tolist()
        elif mode == 'test':
            labs = [1] * len(texts)
        else:
            raise Exception('mode 选择错误！！')
        encodings = self.model.tokenizer(texts, max_length=self.args.max_length, padding='max_length', truncation=True)
        return encodings, labs

    def get_dataloader(self, mode="train") -> DataLoader:
        """get training dataloader"""
        encodings, labs = self.get_encodings(self.args.data_fp, mode)

        dataset = XHSDataset(encodings, labs)

        if mode == "train":
            # define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(2333)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            data_sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset,
                                batch_size=self.args.batch_size,
                                sampler=data_sampler,
                                num_workers=self.args.workers)

        return dataloader

    def test_dataloader(self):
        return self.get_dataloader("test")

    # def test_step(self, batch, batch_idx):
    #     loss, acc = self.compute_loss_and_acc(batch)
    #     return {'test_loss': loss, "test_acc": acc}

    # def test_epoch_end(self, outputs):
    #     test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     test_acc = torch.stack([x['test_acc'] for x in outputs]).mean() / self.num_gpus
    #     tensorboard_logs = {'test_loss': test_loss, 'test_acc': test_acc}
    #     print(test_loss, test_acc)
    #     return {'test_loss': test_loss, 'log': tensorboard_logs}

def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=128, type=int, help="max length of dataset")
    parser.add_argument("--data_fp", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--save_topk", default=2, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proportion", default=0.01, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="dropout probability")

    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = BertTextClassificationTask(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, 'checkpoint', '{epoch}-{val_loss:.4f}-{val_acc:.4f}'),
        save_top_k=args.save_topk,
        save_last=False,
        monitor="val_acc",
        mode="max",
        verbose=True,
        period=-1,
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )

    # save args
    with open(os.path.join(args.save_path, 'checkpoint', "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        del args_dict['gpus']
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger,
                                         deterministic=True)

    trainer.fit(model)


# def evaluate():
#     parser = get_parser()
#     parser = Trainer.add_argparse_args(parser)
#     args = parser.parse_args()
#
#     model = TNewsTask(args)
#     checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['state_dict'])
#     trainer = Trainer.from_argparse_args(args,
#                                          distributed_backend="ddp")
#
#     trainer.test(model)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
