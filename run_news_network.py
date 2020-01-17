#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import time
import pickle as pk
from tqdm import tqdm
from collections import defaultdict

from kistec.preprocess import KistecPreprocess
from kistec.visualize import TopicModeling, WordNetwork

# Data Import
'''
REQUIRE: run_corpus.py
'''

# LDA Modeling
def topic_modeling(target, num_topics):
    fname_prep = './data/preprocessed/news_20191223_v2.pk'
    with open(fname_prep, 'rb') as f:
        corpus = pk.load(f)

    lda_model_config = {
        'corpus': corpus[target],
        'num_topics': num_topics,
        'alpha': 0.7,
        'beta': 0.1
        }

    lda_model = TopicModeling(**lda_model_config)
    lda_model.learn()
    lda_model.assign()
    return lda_model

facility = '교량'
lda_num_topics = 5
fname_lda_result = './result/topic_modeling_v2/lda_model_news_20191223_{}_{}.pk'.format(facility, lda_num_topics)
os.makedirs('./result/topic_modeling_v2/', exist_ok=True)

lda_model = topic_modeling(target=facility, num_topics=lda_num_topics)
with open(fname_lda_result, 'wb') as f:
    pk.dump(lda_model, f)
with open(fname_lda_result, 'rb') as f:
    lda_model = pk.load(f)

# Visualize LDA Model
import pyLDAvis
import pyLDAvis.gensim as gensimvis

def visualize_lda(facility, lda_num_topics, operate=False):
    fname_lda_result = './result/topic_modeling_v2/lda_model_news_20191223_{}_{}.pk'.format(facility, lda_num_topics)
    fname_lda_visual = './result/topic_modeling_v2/lda_visual_news_20191223_{}_{}.html'.format(facility, lda_num_topics)

    if operate:
        pyLDAvis.enable_notebook()
        with open(fname_lda_result, 'rb') as f:
            lda_model = pk.load(f)

        visual_window = gensimvis.prepare(lda_model.model, lda_model.docs_for_lda, lda_model.id2word)
        pyLDAvis.save_html(visual_window, fname_lda_visual)
    else:
        pass

visualize_lda(facility, lda_num_topics, operate=False)

# Word Network
def draw_work_network(lda_model, topic_id, count_option, top_n):
    print('Word Network ...')
    topic_docs = lda_model.docs_by_topic[topic_id]
    stop_list_for_network = ['[0-9]', '[ㄱ-ㅎ]', '등', '것']
    docs_for_network = [[w for w in re.sub('|'.join(stop_list_for_network), '', ' '.join(doc)).split(' ') if len(w)>0] 
        for doc in topic_docs]

    word_network_config = {'docs': docs_for_network,
                           'count_option': count_option,
                           'top_n': top_n,
                           'fname_combs': './data/word_combs_news_20191223_{}_{}_{}'.format(facility, topic_id, count_option),
                           'calculate_combs': True, # 한번 계산한 다음부터는 False로 입력 (데이터가 변하지 않으면 combination 결과도 변하지 않음)
                           'fname_plt': './result/word_network_v2/word_network_news_20191223_{}_{}_{}_{}'.format(facility, topic_id, count_option, top_n)
                           }
    word_network = WordNetwork(**word_network_config)
    word_network.network()

for topic_id in range(lda_num_topics):
    count_option = 'dist'
    for top_n in [50, 100, 150, 200, 500]:
        draw_work_network(lda_model, topic_id, count_option, top_n)