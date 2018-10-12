# -*-coding:utf-8-*-
"""
@Time   : 2018/9/6 16:56
@Author : Mark
@File   : train_sequence.py
"""
import os

from zhuanzhi_ner.data import TaggedCorpus
from zhuanzhi_ner.data.data_fetcher import NLPTaskDataFetcher, NLPTask
from zhuanzhi_ner.trainers.sequence_tagger_trainer import SequenceTaggerTrainer


def load_model_and_train():
    # 训练数据
    train_data_folder = os.path.join('resources', 'corpus', 'data')

    print('加载数据')
    corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(train_data_folder, NLPTask.CONLL_03_ZH)
    print(corpus)

    from zhuanzhi_ner.models import SequenceTagger

    print('载入已有的序列标注模型')
    tagger: SequenceTagger = SequenceTagger.load_from_file("resources/taggers/example-ner/model.pt")

    print('初始化训练器')
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True,
                                                           script_path=u'resources/conll03_eval_script.pl')

    print('开始训练')
    save_model_dir = "resources/taggers/example-ner"
    trainer.train(save_model_dir, learning_rate=0.1, mini_batch_size=32, max_epochs=20, save_model=True)


load_model_and_train()
