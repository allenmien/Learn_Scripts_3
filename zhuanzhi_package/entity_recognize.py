# -*-coding:utf-8-*-
"""
@Time   : 2018/9/4 14:00
@Author : Mark
@File   : entity_recognize.py
"""
from zhuanzhi_ner.data import Sentence
from zhuanzhi_ner.models import SequenceTagger

model_path = u'./resources/tagger/best-model-2.pt'

tagger: SequenceTagger = SequenceTagger.load_from_file(model_path)

sentence = Sentence('国芳集团公司。')
# 预测NER标签
tagger.predict(sentence)
# 输出句句⼦子和相应预测的标签
print(sentence.to_tagged_string())
