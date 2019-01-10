# -*-coding:utf-8-*-
"""
@Time   : 2018/9/10 16:08
@Author : Mark
@File   : __init__.py.py
"""
import json

import numpy as np

es_response = dict()
with open("./response_file.csv", encoding='utf-8') as f:
    es_response = json.load(f)

hits = es_response.get('hits', [])
old_sort_trigger_list = list()
new_sort_trigger_list = list()
for hit in hits[u'hits']:
    old_sort_trigger = dict()
    new_sort_trigger = dict()
    source = hit.get('_source', {})
    score = hit.get('_score')
    if source:
        degree = source.get('taobao_degree', 1000)
        type = source.get('type', 0)
        keyword = source.get('keyword')
        # grade_1 = 0.3 / (1 + np.e ** (-(1.0 * score) / 10))
        grade_2 = 0.6 / (type + 10)
        grade_3 = 0.01 / (1 + np.e ** ((-1 * degree) / 10000))
        grade_1 = score - grade_2 - grade_3

        old_sort_trigger[u'grade_1'] = grade_1
        old_sort_trigger[u'grade_2'] = grade_2
        old_sort_trigger[u'grade_3'] = grade_3
        old_sort_trigger[u'score'] = score
        old_sort_trigger[u'type'] = type
        old_sort_trigger[u'keyword'] = keyword
        old_sort_trigger_list.append(old_sort_trigger)

        new_sort_trigger[u'grade_1'] = grade_1
        new_sort_trigger[u'grade_2'] = 0.6 / (type + 5)
        new_sort_trigger[u'grade_3'] = grade_3
        new_sort_trigger[u'score'] = new_sort_trigger[u'grade_1'] + new_sort_trigger[u'grade_2'] + new_sort_trigger[
            u'grade_3']
        new_sort_trigger[u'type'] = type
        new_sort_trigger[u'keyword'] = keyword
        new_sort_trigger_list.append(new_sort_trigger)

old_sort_trigger_list.sort(key=lambda x: x['score'], reverse=True)
new_sort_trigger_list.sort(key=lambda x: x['score'], reverse=True)

for hit in old_sort_trigger_list:
    print('type:{0}, keyword:{1}, score:{2}, grade_1:{3}, grade_2:{4}, grade_3:{5}'.format(str(hit['type']),
                                                                                           str(hit['keyword']),
                                                                                           str(hit['score']),
                                                                                           str(hit['grade_1']),
                                                                                           str(hit['grade_2']),
                                                                                           str(hit['grade_3'])))
print('----'*50)
for hit in new_sort_trigger_list:
    print('type:{0}, keyword:{1}, score:{2}, grade_1:{3}, grade_2:{4}, grade_3:{5}'.format(str(hit['type']),
                                                                                           str(hit['keyword']),
                                                                                           str(hit['score']),
                                                                                           str(hit['grade_1']),
                                                                                           str(hit['grade_2']),
                                                                                           str(hit['grade_3'])))
