# -*-coding:utf-8-*-
"""
@Time   : 2018/9/6 17:26
@Author : Mark
@File   : preprocess.py
"""

from typing import List
import os

import jieba
from zhuanzhi_ner.data.data_fetcher import NLPTaskDataFetcher, NLPTask
from zhuanzhi_ner.data import TaggedCorpus
from zhuanzhi_ner.data.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, \
    CharacterEmbeddings
from zhuanzhi_ner.data.preprocess import process

from zhuanzhi_package.stopwords import Stopwords


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
        # 分词函数
        word_segmentation_func = jieba.cut
        # 将数据有html格式转化为bioes格式，并进行分词，默认为jieba分词
        process(origin_folder, target_folder, word_segmentation_func=word_segmentation_func)

        # train，test和dev数据文件夹，即训练数据文件夹，无需手动指定
        data_folder = os.path.join(target_folder, 'data')
        # 对bioes格式数据做停用词处理
        stopwords_base_dir = "resources/stopwords"
        stopwords = Stopwords(stopwords_base_dir=stopwords_base_dir)
        stopwords.handle_stopwords(data_folder)


train()
