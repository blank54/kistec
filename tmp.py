#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
from config import Config
import pickle as pk
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from konlpy.tag import Komoran
from datetime import datetime, timedelta
# import pyLDAvis
# import pyLDAvis.gensim as gensimvis
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

matplotlib.rc('font', family='NanumBarunGothic')

from kistec.data_import import *
from kistec.preprocess import KistecPreprocess
from kistec.visualize import TopicModeling, WordNetwork

kp = KistecPreprocess()
kp.build_userdic()
komoran = Komoran(userdic=kp.fname_userdic)

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

facility = '교량'
topic_id = 1
iter_unit = 'month'

fname_docs_topic_assignment = os.path.join(cfg.root, cfg.fdir_topic_assignment, '{}+사고.xlsx'.format(facility))
assigned_data = pd.read_excel(fname_docs_topic_assignment)
date_list = sorted([datetime.strptime(str(date), '%Y%m%d') for date in assigned_data['date']])
_from = date_list[0]
_to = date_list[-1]
_range = _to - _from
if iter_unit == 'month':
    x = list(set([datetime.strftime(_from + timedelta(days=d), '%Y-%m') for d in range(_range.days +1)]))
elif iter_unit == 'date':
    x = list(set([datetime.strftime(_from + timedelta(days=d), '%Y-%m-%d') for d in range(_range.days +1)]))

y_freq = {time:[] for time in x}
y_comment = {time:0 for time in x}
topic_docs = assigned_data.loc[assigned_data['topic_id'] == topic_id]
for idx in range(len(topic_docs)):
    article = topic_docs.iloc[idx]
    if iter_unit == 'month':
        time = datetime.strftime(datetime.strptime(str(article['date']), '%Y%m%d'), '%Y-%m')
    elif iter_unit == 'date':
        time = datetime.strftime(datetime.strptime(str(article['date']), '%Y%m%d'), '%Y-%m-%d')

    y_freq[time].append(article['id'])
    y_comment[time] += int(article['comment_count'])

f1 = plt.figure(num=1, figsize=(40,30))
plt.plot(sorted(x), [len(articles) for time, articles in sorted(y_freq.items(), key=lambda x:x[0])])
plt.title('Article Frequency: {}+사고_{} ({})'.format(facility, topic_id, iter_unit.title()))
plt.xlabel('{}'.format(iter_unit.title()))
plt.ylabel('Frequency')
plt.xticks(rotation=60)
plt.show()