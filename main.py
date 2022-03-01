#%%
import jieba
import pandas as pd
import numpy as np
from model import keyword

df = pd.read_csv('選擇題.csv')

# jieba.set_dictionary('dict.txt.big')  # load extra dictionary

# jieba.load_userdict("userdict.txt")  # load user defined dictionary
# words = jieba.cut(test_string, cut_all=False)  # 使用精確模式
# out = []

# for word in words:
    # out.append(word)
# [list(jieba.cut(i[0], cut_all=False) )for i in  df.values][0]

#%%
# 模型初始化
words = [keyword(i[0]) for i in  df.values]

#%%
words[0].sorting()

#%%
