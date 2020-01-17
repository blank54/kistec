#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import time
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm
from collections import defaultdict

from kistec.data_import import *
from kistec.preprocess import KistecPreprocess
from kistec.visualize import TopicModeling, WordNetwork

'''
REQUIRE: run_corpus.py
'''

# Data Import
def data_import(facility):
    fname_prep = './data/preprocessed/news_20191223_v2.pk'
    with open(fname_prep, 'rb') as f:
        corpus = pk.load(f)
    return corpus[facility]

# Target
facility = '건물'

# LDA Tuning
def lda_tuning(facility):
    corpus = data_import(facility)
    target_docs = [(article.id, article.content_prep) for article in corpus]

    lda_tuning_config = {
        'corpus': target_docs,
        'min_topics': 3,
        'max_topics': 10,
        'topics_step': 1,
        'min_alpha': 0.1,
        'max_alpha': 1.0,
        'alpha_step': 0.3,
        'alpha_symmetric': False,
        'alpha_asymmetric': False,
        'min_beta': 0.1,
        'max_beta': 1.0,
        'beta_step': 0.3,
        'beta_symmetric': False
        }
    lda_model = TopicModeling(**lda_tuning_config)
    lda_tuning_result = lda_model.tuning()
    return lda_tuning_result

print('LDA Tuning: {}'.format(facility))
fname_lda_tuning_result = './result/topic_modeling_v2/lda_tuning_news_20191223_{}.xlsx'.format(facility)
lda_tuning_result = lda_tuning(facility)
save_df2excel(lda_tuning_result, fname_lda_tuning_result, verbose=True)
lda_tuning_result = pd.read_excel(fname_lda_tuning_result)

# LDA Modeling
def get_optimum_parameters(lda_tuning_result):
    max_coherence = lda_tuning_result.sort_values(by='Coherence', ascending=False).iloc[0]

    lda_num_topics = int(max_coherence['Num_of_Topics'])
    alpha = np.around(max_coherence['Alpha'], decimals=1)
    beta = np.around(max_coherence['Beta'], decimals=1)

    return lda_num_topics, alpha, beta

def topic_modeling(facility, lda_tuning_result):
    lda_num_topics, alpha, beta = get_optimum_parameters(lda_tuning_result)
    print('LDA Modeling: {} / {}-{}-{}'.format(facility, lda_num_topics, alpha, beta))
    corpus = data_import(facility)
    target_docs = [(article.id, article.content_prep) for article in corpus]

    lda_model_config = {
        'corpus': target_docs,
        'num_topics': lda_num_topics,
        'alpha': alpha,
        'beta': beta
        }

    lda_model = TopicModeling(**lda_model_config)
    lda_model.learn()
    lda_model.assign()
    return lda_model

lda_model = topic_modeling(facility, lda_tuning_result)
lda_num_topics = lda_model.num_topics
fname_lda_model = './model/lda_model_news_20191223_{}_{}.pk'.format(facility, lda_num_topics)
with open(fname_lda_model, 'wb') as f:
    pk.dump(lda_model, f)
with open(fname_lda_model, 'rb') as f:
    lda_model = pk.load(f)

# Topic Assignment
def topic_assignment(lda_model, facility, fname_docs_topic):
    print('Topic Assignment: {}'.format(fname_docs_topic))
    fdir = '/'.join(fname_docs_topic.split('/')[:-1])
    os.makedirs(fdir, exist_ok=True)

    corpus = data_import(facility)
    articles = []
    for article in corpus:
        article.topic_id = lda_model.tag2topic[article.id]
        articles.append(article)

    df = articles2df(articles)
    save_df2excel(df, fname_docs_topic, verbose=True)

fname_docs_topic = './data/topic/news_20191223_{}_{}.xlsx'.format(facility, lda_num_topics)
topic_assignment(lda_model, facility, fname_docs_topic)

# Visualization
import pyLDAvis
import pyLDAvis.gensim as gensimvis

def visualize_lda(lda_model, facility):
    visual_window = gensimvis.prepare(lda_model.model, lda_model.docs_for_lda, lda_model.id2word)
    fname_lda_visual = './result/topic_modeling_v2/lda_visual_news_20191223_{}_{}.html'.format(facility, lda_model.num_topics)
    pyLDAvis.save_html(visual_window, fname_lda_visual)

visualize_lda(lda_model, facility)