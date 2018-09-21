# -*-coding:utf-8-*-
"""
@Time   : 2018/9/21 9:12
@Author : Mark
@File   : predict_sentence.py
"""
from zhuanzhi_ner.data import Sentence
from zhuanzhi_ner.predictions.predict import Predict

model_path = u'./resources/tagger/best-model-2.pt'
predict = Predict(model_path)
source_dir = "./resources/data/source"
target_dir = "./resources/data/target"
# 遍历source_dir的所有⽂文件并对内容进⾏行行回标，标注后的内容保存在target_dir下的同名⽂文件中
predict.tag_file(source_dir, target_dir)
