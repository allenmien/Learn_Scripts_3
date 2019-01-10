# -*-coding:utf-8-*-
import pandas as pd
import jieba
from jieba import analyse

jieba.load_userdict("new_words_dict.csv")

tfidf = analyse.extract_tags
# textrank = analyse.textrank
#
# data = pd.read_csv('./searchword.csv', encoding='utf-8', )
# data.columns = ['search_word', 'search_count', 'recall', 'search_data']
# tf_keywords = list()
# tr_keywords = list()
# for index, row in data.head(20).iterrows():
#     search_word = row['search_word']
#     tf_keyword = tfidf(search_word, allowPOS=['n'])
#     tr_keyword = textrank(search_word)
#     tf_keywords.append(tf_keyword)
#     tr_keywords.append(tr_keyword)
# tf_keywords_column = {'tf_keywords': tf_keywords}
# tr_keywords_column = {'tr_keywords': tr_keywords}
# keyword_data = pd.concat(
#     [data['search_word'].head(20), pd.DataFrame(tf_keywords_column), pd.DataFrame(tr_keywords_column)], axis=1)
print(' '.join(jieba.cut('三只松鼠 手剥巴旦木235gx2袋 零食坚果炒货特产干果巴达木')))
tf_keyword = tfidf('三只松鼠 手剥巴旦木235gx2袋 零食坚果炒货特产干果巴达木')
print(' '.join(tf_keyword))
