# -*- coding: utf-8 -*-
# @Time    : 3/6/23 5:25 PM
# @Author  : LIANYONGXING
# @FileName: predict_script.py.py
from task.text_classification_train_task import BertTextClassificationTask
import pytorch_lightning as pl
import pandas as pd


if __name__ == '__main__':

    test_filepath = '/Users/user/Downloads/final_train_v1.csv'

    bert = BertTextClassificationTask.load_from_checkpoint(
        'new_version2/epoch=0-step=20.ckpt',
    )

    test_dl = bert.get_test_dataloader(path=test_filepath, max_length=256, batch_size=8)

    bert.eval()
    trainer = pl.Trainer()
    predictions = trainer.predict(bert, dataloaders=test_dl, return_predictions=True)

    scores = [i[1][:, 1].tolist() for i in predictions]
    all_scores = sum(scores, [])

    raw_test_datas = pd.read_csv('/Users/user/Downloads/final_train_v1.csv')[:20]
    raw_test_datas['score'] = all_scores
    raw_test_datas.to_csv('result1.csv', index=False, encoding='utf_8_sig')

