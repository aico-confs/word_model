#%%
import jieba
import pandas as pd
import numpy as np
from model import keyword

df = pd.read_csv('選擇題.csv')



#%%
# 模型初始化
words = [keyword(i[0]) for i in  df.values]

#%%
words[4].sorting()

#%%
