#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import pickle as pk
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from konlpy.tag import Komoran

from kistec.data_import import *
from kistec.preprocess import KistecPreprocess

preprocess_config = {
    'fname_userword': './corpus/thesaurus/userword.txt',
    'fname_userdic': './corpus/thesaurus/userdic.txt',
    'fname_stop': './corpus/thesaurus/stop_list.txt',
    'fname_synonyms': './corpus/thesaurus/synonyms.txt'
    }
kp = KistecPreprocess(**preprocess_config)
kp.build_userdic()
komoran = Komoran(userdic=kp.fname_userdic)

'''
REQUIRE: run_news_crawling_v2.py
'''

# Merge Docs
def merge_docs(fdir_data):
    print('Merge Docs ...')
    flist = [f for f in os.listdir(fdir_data) if f.endswith('.xlsx')]

    corpus = defaultdict(list)
    with tqdm(total=len(flist)) as pbar:
        for fname in flist:
            facility = fname.split('_')[1].split('+')[0]
            data = pd.read_excel(os.path.join(fdir_data, fname))
            for idx in range(len(data)):
                current_article = data.iloc[idx]
                article_config = {
                    'url':current_article['url'],
                    'title':current_article['title'],
                    'date':current_article['date'],
                    'category':current_article['category'],
                    'content':current_article['content'],
                    'comment_list':current_article['comment_list'],
                    'comment_count':current_article['comment_count']
                    }
                corpus[facility].append(Article(**article_config))
            pbar.update(1)

    for facility in corpus:
        print('{}: {}'.format(facility, len(corpus[facility])))        
    return corpus

fdir_data = '/data/blank54/workspace/kistec_workspace/data/news/20191223/'
fname_docs_raw = './corpus/corpus_news_20191223.pk'
# corpus = merge_docs(fdir_data)
# with open(fname_docs_raw, 'wb') as f:
#     pk.dump(corpus, f)
with open(fname_docs_raw, 'rb') as f:
    corpus = pk.load(f)

# Drop Duplicates
def drop_duplicates(corpus):
    corpus_drop_duplicate = defaultdict(list)
    for facility in corpus:
        print('Drop Duplicates: {}'.format(facility))
        _url_list = []
        idx = 0
        with tqdm(total=len(corpus[facility])) as pbar:
            for article in corpus[facility]:
                if article.url not in _url_list:
                    article.id = '{}_{}'.format(facility, idx)
                    corpus_drop_duplicate[facility].append(article)
                    _url_list.append(article.url)
                    idx += 1
                else:
                    continue
                pbar.update(1)

    for facility in corpus_drop_duplicate:
        print('{}: {}'.format(facility, len(corpus_drop_duplicate[facility])))
    return corpus_drop_duplicate

fname_corpus = './corpus/corpus_news_20191223_drop_duplicate.pk'
# corpus_drop_duplicate = drop_duplicates(corpus)
# with open(fname_corpus, 'wb') as f:
#     pk.dump(corpus_drop_duplicate, f)
with open(fname_corpus, 'rb') as f:
    corpus_drop_duplicate = pk.load(f)

# Preprocessing
def mark_needless(text):
    with open('./corpus/thesaurus/needless_list.txt', 'r') as f:
        needless_list = f.read().replace('\n', '|')
    text = re.sub(needless_list, ' NEEDLESS ', text)
    return text

def drop_needless(text):
    marked_sent = kp.tokenize(mark_needless(text))
    new_sent = []
    needless_index_list = [i for i, w in enumerate(marked_sent) if w == 'NEEDLESS']

    if needless_index_list:
        for idx in range(len(needless_index_list)):
            if idx == 0:
                new_sent.extend(marked_sent[:needless_index_list[idx]])
            else:
                i = needless_index_list[idx-1]
                j = needless_index_list[idx]
                if (j-i) < 5:
                    pass # NEEDLESS의 간격이 5단어 미만이면 -> 이건 광고!
                else:
                    new_sent.extend(marked_sent[i:j])

            if idx == (len(needless_index_list)-1):
                if (len(marked_sent) - needless_index_list[idx]) > 10: # 마지막 NEEDLESS 이후 10단어보다 많이 남았으면
                    new_sent.extend(marked_sent[needless_index_list[idx]:])
                else:
                    continue
    else:
        new_sent = marked_sent

    return ' '.join([w for w in new_sent if not w == 'NEEDLESS'])

def preprocess(corpus):
    corpus_prep = defaultdict(list)
    for facility in corpus_drop_duplicate:
        print('PoS Tagging (get nouns): {}'.format(facility))
        df_facility = defaultdict(list)
        with tqdm(total=len(corpus_drop_duplicate[facility])) as pbar:
            for article in corpus_drop_duplicate[facility]:
                content_clean = drop_needless(article.content)
                if content_clean:
                    content_nouns = komoran.nouns(content_clean)
                    content_nouns_stop = kp.stopword_removal(' '.join(content_nouns))
                    article.content_prep = content_nouns_stop
                    if content_nouns_stop:
                        corpus_prep[facility].append(article)
                        df_facility = articles2df(corpus_prep[facility])
                    else:
                        continue
                else:
                    continue
                pbar.update(1)
        
        save_df2excel(
            data=pd.DataFrame(df_facility),
            fname='./corpus/preprocessed/news/news_20191223_v2_{}.xlsx'.format(facility),
            verbose=True)

    return corpus_prep

fname_prep = './corpus/preprocessed/news/news_20191223_v2.pk'
corpus_prep = preprocess(corpus_drop_duplicate)
with open(fname_prep, 'wb') as f:
    pk.dump(corpus_prep, f)
with open(fname_prep, 'rb') as f:
    corpus_prep = pk.load(f)