#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import pickle as pk
import pandas as pd
from config import Config
from tqdm import tqdm
from collections import defaultdict
from konlpy.tag import Komoran

from kistec.object import *
from kistec.function import *

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

fdir_news_articles = os.path.join(cfg.root, cfg.fdir_news_articles)

fdir_old = '/data/blank54/workspace/kistec_workspace/data/news/20191223/'
flist = [f for f in os.listdir(fdir_old) if f.endswith('.xlsx')]

print('Refactorying ...')
with tqdm(total=len(flist)) as pbar:
    for fname in flist:
        data = pd.read_excel(os.path.join(fdir_old, fname))
        for idx in range(len(data)):
            _article = data.iloc[idx]

            article_config = {
                'url': _article['url'],
                'title': _article['title'],
                'date': _article['date'],
                'category': _article['category'],
                'content': _article['content'],
                'comment_list': _article['comment_list'],
                'comment_count': _article['comment_count']
                }
            article = Article(**article_config)

            fname_article = os.path.join(fdir_news_articles, article.date, '{}.pk'.format(article.url_uniq))
            makedir(fname_article)
            with open(fname_article, 'wb') as f:
                pk.dump(article, f)
        pbar.update(1)

print('done!!!')