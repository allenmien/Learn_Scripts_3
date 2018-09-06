# -*-coding:utf-8-*-
"""
@Time   : 2018/9/6 17:26
@Author : Mark
@File   : preprocess.py
"""

from typing import List
import os
from zhuanzhi_ner.data.data_fetcher import NLPTaskDataFetcher, NLPTask
from zhuanzhi_ner.data import TaggedCorpus
from zhuanzhi_ner.data.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, \
    CharacterEmbeddings
from zhuanzhi_ner.data.preprocess import process


def train():
    '''
    测试训练序列标注模型
    :return:
    '''

    re_precess_data = True

    # 预处理数据
    if re_precess_data:
        # 需要数据预处理源文件夹
        origin_folder = os.path.join('resources', 'origin_corpus', 'ner_corpus')
        # 处理后数据存储的目标文件夹
        target_folder = os.path.join('resources', 'corpus')

        # 指定对已经分好词的语料文本进行预处理
        use_word_segmented_corpus = True
        # 指定文本中词之间的分割符
        word_separator = " "
        # 开始数据预处理
        process(origin_folder, target_folder, use_word_segmented_corpus=use_word_segmented_corpus,
                word_separator=word_separator)


train()
