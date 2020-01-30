#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import pickle as pk
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from config import Config
with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

class Article:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', '')
        self.url = kwargs.get('url', '')
        self.url_uniq = self.url.split('/')[-1]
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