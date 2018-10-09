# !/usr/bin/python3
# -*- coding:utf8 -*-
"""
@Time   : 2018/9/29 16:50
@Author : Mark
@File   : aa.py
"""
# import codecs

lines = list()
path_to_column_file = u'./corpus/data/zh.train'
# with open(u'./corpus/data/zh.train', 'rb') as f:
#     for l in f.readlines():
#         lines.append(l)

lines = open(path_to_column_file, 'rb').read().decode(u'utf-8').strip().split('\n')
print(lines)
