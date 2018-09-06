# -*-coding:utf-8-*-
"""
@Time   : 2018/9/6 16:56
@Author : Mark
@File   : train_sequence.py
"""
from zhuanzhi_ner.data import TaggedCorpus
from zhuanzhi_ner.data.data_fetcher import NLPTaskDataFetcher, NLPTask

data_folder = './resources/corpora/data'
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(data_folder, NLPTask.CONLL_03_ZH)
original_corpus = NLPTaskDataFetcher.fetch_data(data_folder, NLPTask.CONLL_03_ZH)
downsampled_corpus = original_corpus.downsample(0.1)

print("--- 1 Original ---")
print(original_corpus)
print("--- 2 Downsampled ---")
print(downsampled_corpus)
