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

from kistec.object import *
from kistec.function import *
from kistec.visualize import *

kp = KistecPreprocess()
kp.build_userdic()
komoran = Komoran(userdic=kp.fname_userdic)

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)







# from wordcloud import WordCloud

query = '터널+사고'
# fname_pathtofile = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_{}_after2015.pk'.format(query))
fname_pathtofile = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_prep_{}_after2015.pk'.format(query))
with open(fname_pathtofile, 'rb') as f:
    pathtofile = pk.load(f)



corpus = []
for fname in pathtofile:
    with open(fname, 'rb') as f:
        corpus.append(pk.load(f))

print(len(corpus))
for article in corpus:
    print(article.id)