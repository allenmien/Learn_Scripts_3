# -*-coding:utf-8-*-
import pandas as pd

file_path = './query_corpus.csv'
data = pd.read_csv('./searchword.csv', encoding='utf-8', )
data.columns = ['search_word', 'search_count', 'recall', 'search_data']
data['search_word'].to_csv(file_path, encoding='utf-8', index=False)
