# -*-coding:utf-8-*-
"""
@Time   : 2018/10/9 14:53
@Author : Mark
@File   : test_sequence_model.py
"""
import os
import sys
import threading

from zhuanzhi_ner.data import TaggedCorpus
from zhuanzhi_ner.data.data_fetcher import NLPTaskDataFetcher, NLPTask
from zhuanzhi_ner.data.embeddings import TokenEmbeddings, WordEmbeddings, CharLMEmbeddings, StackedEmbeddings
from typing import List


class EvaluateSequenceModel(object):
    def __init__(self):
        pass

    def process(self):
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
            # WordEmbeddings('resources/Word_model/sgns.financial.char', corpus),
            WordEmbeddings('resources/Word_model/sgns.financial.word', corpus),
            # WordEmbeddings('resources/Word_model/sgns.sogou.char', corpus),
            # WordEmbeddings('resources/Word_model/sgns.sogou.word', corpus),
            # WordEmbeddings('resources/Word_model/ns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5', corpus),
            # WordEmbeddings('resources/Word_model/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', corpus),
            CharLMEmbeddings('resources/LM_model/best-lm.pt')
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
                                                               test_mode=True,
                                                               script_path=u'resources/conll03_eval_script.pl'
                                                               )
        # 7. 开始训练
        trainer.evaluate(corpus.dev,
                         'resources/taggers/example-ner',
                         evaluation_method='span-F1',
                         embeddings_in_memory=True)


EvaluateSequenceModel().process()
