# -*-coding:utf-8-*-
"""
@Time   : 2018/9/5 17:22
@Author : Mark
@File   : lm_model_test.py
"""
import os
from zhuanzhi_ner.data import Sentence
from zhuanzhi_ner.data.embeddings import CharLMEmbeddings
os.environ['CUDA_VISIBLE_DEVICES'] = ''
model_path = u'./resources/LM_model/lm-1.pt'

# 实例例化CharLMEmbeddings类
charlm_embedding_forward = CharLMEmbeddings(model_path)
# 创建`Sentence`对象
sentence = Sentence('国防集团 公司 .')
# embed句句⼦子中的词
sentence_embedding = charlm_embedding_forward.embed(sentence)