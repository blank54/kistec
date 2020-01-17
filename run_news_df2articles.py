#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
from kistec.web_crawling import *

import os
import pickle as pk
import pandas as pd
from tqdm import tqdm

fdir = './articles_20191223/'
flist = [f for f in os.listdir(fdir) if f.endswith('.pk')]
with tqdm(total=len(flist)) as pbar:
    for fname in flist:
        df2articles(os.path.join(fdir, fname))
        pbar.update(1)