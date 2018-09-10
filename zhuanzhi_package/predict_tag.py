# -*-coding:utf-8-*-
# @Time    : 18/9/10 09:47
# @Author  : Mark
# @File    : predict_tag.py

from typing import List
import os
from zhuanzhi_ner.predictions.predict import Predict
import jieba

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def test_predict_tag():
    '''
        给定sentence，对其NER标签进行预测
        :return:
    '''

    # 预训练模型路径
    model_path = "resources/taggers/fenci-ner-1/2/model-25.pt"

    # 停用词路径
    stopwords_base_dir = "resources/stopwords"
    # 分词函数
    # word_segmentation_func = jieba.cut
    # 使用指定的分词函数word_segmentation_func初始化NER标签预测器
    # predict = Predict(model_path, stopwords_base_dir, word_segmentation_func=word_segmentation_func)

    # 指定对已经分好词的语料文本进行预测
    use_word_segmented_corpus = True
    # 指定文本中词之间的分割符
    word_separator = " "
    # 初始化预测器
    predict = Predict(model_path, use_word_segmented_corpus=use_word_segmented_corpus, word_separator=word_separator)

    # 测试文本
    s = "国芳 集团 (601086) 公司简介 公司 从事 以 百货业 为主 。"

    # 预测NER标签，返回Sentence对象
    sentence = predict.predict_tags(s)

    # 输出预测结果
    print(sentence.to_tagged_string())
    print(predict.get_tagged_chinese_string(sentence))
