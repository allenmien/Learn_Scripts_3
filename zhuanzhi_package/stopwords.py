# coding=utf8
from typing import List

import os
import re


class Stopwords(object):
    def __init__(self, stopwords_base_dir):
        '''
        对数据做停用词处理
        :param stopwords_base_dir: the directory of standard stopword files
        '''

        self.stopwords_base_dir = stopwords_base_dir
        print("loading stop words ....")
        self.stopword_list = self._load_stopword_list()

    def _load_stopword_list(self):
        stopword_list = []
        for name in os.listdir(self.stopwords_base_dir):
            path = os.path.join(self.stopwords_base_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                swords = [line.replace("\n", "") for line in f]
            stopword_list.extend(swords)
        return list(set(stopword_list))

    def handle_stopwords(self, data_folder):
        for name in os.listdir(data_folder):
            file_path = os.path.join(data_folder, name)
            bak_file_path = os.path.join(data_folder, "{}.bak".format(name))
            os.rename(file_path, bak_file_path)
            with open(bak_file_path, "r", encoding="utf-8") as f:
                with open(file_path, "w+", encoding="utf-8") as f_write:
                    for line in f:
                        if line.strip() == "":
                            f_write.write(line)
                            continue
                        fields: List[str] = re.split("\s+", line)
                        label = fields[3]
                        word = fields[0]
                        if word in self.stopword_list:
                            f_write.write("<unk> _ _ {}\n".format(label))
                        else:
                            f_write.write(line)
            os.remove(bak_file_path)
