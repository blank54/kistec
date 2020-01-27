#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import pickle as pk
import pandas as pd
from config import Config
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from konlpy.tag import Komoran

from kistec.object import *
from kistec.function import *

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

kp = KistecPreprocess()
kp.build_userdic()
komoran = Komoran(userdic=kp.fname_userdic)

# Corpus Initialization
def init_corpus(query):
    print('Init Corpus ({}) ...'.format(query))
    fname_pathtofile = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_{}_after2015.pk'.format(query))

    with open(fname_pathtofile, 'rb') as f:
        pathtofile = pk.load(f)

    with tqdm(total=len(pathtofile)) as pbar:
        for fname in pathtofile:
            with open(fname, 'rb') as f:
                article = pk.load(f)

            article.id = ''
            article.content_prep = ''
            article.topic_id = ''

            with open(fname, 'wb') as f:
                pk.dump(article, f)
            pbar.update(1)

# Build Corpus
def build_corpus(query):
    print('Build Corpus ({}) ...'.format(query))
    fname_pathtofile = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_{}_after2015.pk'.format(query))
    fdir_news_articles = os.path.join(cfg.root, cfg.fdir_news_articles)

    pathtofile = []
    dlist = []
    for dname in os.listdir(fdir_news_articles):
        try:
            year = datetime.strptime(dname, '%Y%m%d').year
        except ValueError:
            continue
        if year >= 2015:
            dlist.append(dname)

    with tqdm(total=len(dlist)) as pbar:
        for dname in dlist:
            for fname in os.listdir(os.path.join(fdir_news_articles, dname)):
                filepath = os.path.join(fdir_news_articles, dname, fname)
                with open(filepath, 'rb') as f:
                    article = pk.load(f)
                if all((q in article.content for q in query.split('+'))):
                    pathtofile.append(filepath)
            pbar.update(1)

    with open(fname_pathtofile, 'wb') as f:
        pk.dump(pathtofile, f)
    print(len(pathtofile))

# Preprocess
def append_article2df(article, df):
    df['id'].append(article.id)
    df['date'].append(article.date)
    df['title'].append(article.title)
    df['category'].append(article.category)
    df['content'].append(article.content)
    df['content_prep'].append(article.content_prep)
    df['comment_list'].append(article.comment_list)
    df['comment_count'].append(article.comment_count)
    df['url'].append(article.url)
    return df

def preprocess(query):
    print('Preprocess ({}) ...'.format(query))
    fname_pathtofile = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_{}_after2015.pk'.format(query))
    fname_pathtofile_prep = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_prep_{}_after2015.pk'.format(query))
    fname_corpus_prep = os.path.join(cfg.root, cfg.fdir_news_prep, '{}_after2015.xlsx'.format(query))

    with open(fname_pathtofile, 'rb') as f:
        pathtofile = pk.load(f)

    pathtofile_prep = []    
    _errors_clean = []
    _errors_nouns = []
    
    df = defaultdict(list)
    idx = 0
    with tqdm(total=len(pathtofile)) as pbar:
        for fname_article in pathtofile:
            with open(fname_article, 'rb') as f:
                article = pk.load(f)

            content_clean = kp.drop_needless(article.content)
            if content_clean.strip():
                content_prep = ' '.join(kp.stopword_removal(' '.join(komoran.nouns(content_clean))))
                if content_prep:
                    article.content_prep = content_prep
                    article.id = '{}_{}'.format(query, idx)
                    
                    fname_article_prep = os.path.join(cfg.root, cfg.fdir_news_prep, query, os.path.basename(fname_article))
                    makedir(fname_article_prep)
                    pathtofile_prep.append(fname_article_prep)
                    df = append_article2df(article, df)
                    with open(fname_article_prep, 'wb') as f:
                        pk.dump(article, f)
                    idx += 1
                else:
                    _errors_nouns.append(fname_article)
                    pass
            else:
                _errors_clean.append(fname_article)
                pass
            pbar.update(1)

    with open(fname_pathtofile_prep, 'wb') as f:
        pk.dump(pathtofile_prep, f)
    save_df2excel(pd.DataFrame(df), fname_corpus_prep)

    print('pathtofile: {}'.format(len(pathtofile)))
    print('pathtofile_prep: {}'.format(len(pathtofile_prep)))
    print('df: {}'.format(len(df)))
    print('_errors_clean: {}'.format(len(_errors_clean)))
    print('_errors_nouns: {}'.format(len(_errors_nouns)))

# main
def main():
    query_list = ['교량+사고', '건물+사고', '터널+사고']
    for query in query_list:
        init_corpus(query)
        build_corpus(query)
        preprocess(query)

main()