import jieba
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import sys, os
def cutword(txt):
    txt = re.sub('\s', '', txt)
    my_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    path = os.path.join(my_path, 'dict.txt')
    jieba.set_dictionary(path)
    path = os.path.join(my_path, "dict.txt.big")
    jieba.load_userdict(path)
    cut_result = jieba.cut(txt, cut_all=False, HMM=True)
    cut_word = list(cut_result)
    
    # 句子
    sentences = re.findall(r'.*?[，。、：,.\s]' ,txt)
    sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
    cut_sentence = [" ".join(sent0) for sent0 in sent_words]
    return cut_word,cut_sentence
def clearing(cut):        
        with open('停用詞-繁體中文.txt', 'r', encoding="utf-8") as file:
            stop_words = file.readlines()
        with open('special_stop.txt', 'r', encoding="utf-8") as file:
            stop_words += file.readlines()
        # print(len(stop_words))
        stop_words = [word.strip('\n') for word in stop_words] 
        unsorted_word = [word for word in cut if not word in set(stop_words)]
        return unsorted_word
class keyword():
    def __init__(self, txt:str):
        self.txt = txt.strip('\n').strip(' ').replace('\n','')
        self.cutword = cutword(self.txt)[0]
        self.cutsentence = cutword(self.txt)[1]
        self.clearword = clearing(self.cutword)
    
# https://blog.csdn.net/blmoistawinde/article/details/80816179
    def sorting(self):
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
        cv=CountVectorizer(max_features=1000, token_pattern=r"(?u)\b\w\w+\b")
        sentences = re.findall(r'.*?[，。、：,.\s]' ,self.txt)
        if not sentences:
            raise ValueError('配對為空')
        sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
        document = [" ".join(sent0) for sent0 in sent_words]
        
        sk_tfidf = vectorizer.fit_transform(document)
        bag = cv.fit_transform(document)
        dict_ = vectorizer.vocabulary_
        all_word = cv.get_feature_names()
        idf = abs((vectorizer.idf_ - np.mean(vectorizer.idf_))/np.std(vectorizer.idf_))

        key = cv.transform(self.clearword)
        sum_list = np.sum(key.toarray(), axis=0)*idf
        return sorted([i for i in list(zip(all_word, sum_list)) if i[1]], key = lambda x :x[1], reverse=True)


# sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
#         document = [" ".join(sent0) for sent0 in sent_words]
        
#         sk_tfidf = vectorizer.fit_transform(document)
#         bag = cv.fit_transform(document)
#         dict_ = vectorizer.vocabulary_



# print(word.cut)

# print(word.clearword)
# print(word.sorting())