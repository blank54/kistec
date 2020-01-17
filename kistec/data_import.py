#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import pickle as pk
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def make_dir(fname):
    if fname.split('/')[:-1]:
        fdir = '/'.join(fname.split('/')[:-1])
        os.makedirs(fdir, exist_ok=True)

def save_df2excel(data, fname, verbose=False):
    make_dir(fname)
        
    writer = pd.ExcelWriter(fname)
    data.to_excel(writer, "Sheet1", index=False)
    writer.save()
    if verbose:
        print("Saved data as: {}".format(fname))

def articles2df(articles):
    data = defaultdict(list)
    for article in articles:
        data['id'].append(article.id)
        data['content'].append(article.content)
        data['content_nouns'].append(article.content_prep)
        data['comment_list'].append(article.comment_list)
        data['comment_count'].append(article.comment_count)
        data['url'].append(article.url)
        data['topic_id'].append(article.topic_id)
    return pd.DataFrame(data)

class Article:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', '')
        self.url = kwargs.get('url', '')
        self.title = kwargs.get('title', '')
        self.date = kwargs.get('date', '')
        self.category = kwargs.get('category', '')
        self.content = kwargs.get('content', '')
        self.content_prep = kwargs.get('content_prep', '')

        self.likeit_good = kwargs.get('likeit_good', '')
        self.likeit_warm = kwargs.get('likeit_warm', '')
        self.likeit_sad = kwargs.get('likeit_sad', '')
        self.likeit_angry = kwargs.get('likeit_angry', '')
        self.likeit_want = kwargs.get('likeit_want', '')

        self.comment_list = kwargs.get('comment_list', 'none')
        self.comment_count = kwargs.get('comment_count', 0)

        self.topic_id = kwargs.get('topic_id', '')