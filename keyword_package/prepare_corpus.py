# -*-coding:utf-8-*-
import json

import pandas as pd
import redis


class Processer(object):
    def __init__(self):
        self.host = ''
        self.port = ''
        self.password = ''
        self.pool = redis.ConnectionPool(host=self.host, password=self.password)
        self.redis_connection = redis.Redis(connection_pool=self.pool)

    def start(self):
        young_mall_cats_cache_str = self.redis_connection.get('young_mall_cats_cache').decode('utf-8')
        young_mall_cats_cache_list = json.loads(young_mall_cats_cache_str)
        word_list = list()
        for cache_item in young_mall_cats_cache_list:
            words = cache_item.split('/')
            word_list.extend([word.strip() for word in words if word.strip()])
        redis_keys = ['young_mall_brand_area_coupon_name_cache', 'young_mall_search_keywords_hot_cache']
        for redis_key in redis_keys:
            cache_str = self.redis_connection.get(redis_key).decode('utf-8')
            young_mall_brand_area_coupon_name_cache_list = json.loads(cache_str)
            word_list.extend([word.strip() for word in young_mall_brand_area_coupon_name_cache_list if word.strip()])
        data = pd.DataFrame(list(set(word_list)))
        data.to_csv('words.csv', index=False, encoding='utf-8', index_label=False)
        pass


if __name__ == '__main__':
    Processer().start()
