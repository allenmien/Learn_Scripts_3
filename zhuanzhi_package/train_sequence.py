# -*-coding:utf-8-*-
"""
@Time   : 2018/9/6 16:56
@Author : Mark
@File   : train_sequence.py
"""
import os

from zhuanzhi_ner.data import TaggedCorpus
from zhuanzhi_ner.data.data_fetcher import NLPTaskDataFetcher, NLPTask
from zhuanzhi_ner.data.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List


def train():
    '''
    测试训练序列标注模型
    :return:
    '''

    origin_folder = os.path.join('resources', 'origin_corpus',
                                 'ner_corpus')
    target_folder = os.path.join('resources', 'corpus')
    data_folder = os.path.join('resources', 'corpus', 'data')

    # 2. 指定预测标签
    tag_type = 'ner'
    # 3. 从语料料库中创建标签字典
    corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(data_folder, NLPTask.CONLL_03_ZH)  # .downsample(0.1)
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)
    # 4. 实例例化embeddings
    print('初始化词向量')
    embedding_types: List[TokenEmbeddings] = [
        CharLMEmbeddings('resources/LM_model/best-lm.pt'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    # 5. 初始化序列列标记器器
    from zhuanzhi_ner.models import SequenceTagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)
    # 6. 初始化模型训练器器
    from zhuanzhi_ner.trainers import SequenceTaggerTrainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus,
                                                           test_mode=True)
    # 7. 开始训练
    trainer.train('resources/taggers/example-ner',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150)


train()
