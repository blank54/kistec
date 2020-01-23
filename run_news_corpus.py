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

from kistec.data_import import *
from kistec.preprocess import KistecPreprocess

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

kp = KistecPreprocess()
kp.build_userdic()
komoran = Komoran(userdic=kp.fname_userdic)

'''
REQUIRE: run_news_crawling.py
'''





# Build Corpus
def build_corpus(queries):
    print('Build Corpus: {}'.format(queries))

    fdir_news_articles = os.path.join(cfg.root, cfg.fdir_news_articles)
    fname_pathtofile = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_{}.pk'.format(queries))
    pathtofile = []
    with tqdm(total=len(fdir_news_articles)) as pbar:
        for fname in fdir_news_articles:
            with open(os.path.join(fdir_news_articles, fname), 'rb') as f:
                article = pk.load(f)
            if all((query in article.content for query in queries.split('+'))):
                pathtofile.append(fname)
            pbar.update(1)

    with open(fname_pathtofile, 'wb') as f:
        pk.dump(pathtofile, f)
    print(len(len(fdir_news_articles)))
    print(len(pathtofile))

# Preprocess
def preprocess(queries):
    fname_pathtofile = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_{}.pk'.format(queries))
    fname_pathtofile_prep = os.path.join(cfg.root, cfg.fdir_news_pathtofile_prep, 'pathtofile_prep_{}.pk'.format(queries))

    with open(fname_pathtofile, 'rb') as f:
        pathtofile = pk.load(f)
    _errors_clean = []
    _errors_nouns = []

    with tqdm(total=len(pathtofile)) as pbar:
        for fname_article in pathtofile:
            with open(fname_article, 'rb') as f:
                article = pk.load(f)
                fname_article_prep = os.path.join(cfg.root, cfg.fdir_news_articles_prep, '{}.pk'.format(article.url_uniq))

            content_clean = kp.drop_needless(article.content)
            if content_clean:
                content_prep = kp.stopword_removal(' '.join(komoran.nouns(content_clean)))
                if content_prep:
                    article.content_prep = content_prep
                else:
                    _errors_nouns.append(fname_article)
                    continue
            else:
                _errors_clean.append(fname_article)
                continue

            with open(fname_article_prep, 'wb') as f:
                pk.dump(article, f)
            pbar.update(1)

    with open(fname_pathtofile_prep, 'wb') as f:
        pk.dump(pathtofile_prep, f)
    print(len(pathtofile))
    print(len(pathtofile_prep))
    print(_errors_clean)
    print(len(_errors_clean))
    print(_errors_nouns)
    print(len(_errors_nouns))

def main():
    queries = '교량+유지관리+사고'
    build_corpus(queries)
    preprocess(queries)