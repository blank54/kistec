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

# Target
facility = '교량'
lda_num_topics = 5
# topic_id = 0 # Topic Map에서 1번 --> Topic ID: 0번

def main(facility, lda_num_topics, topic_id):
    # Data Import
    def model_import(facility, lda_num_topics):
        fname_lda_result = './model/lda_model_news_20191223_{}_{}.pk'.format(facility, lda_num_topics)
        with open(fname_lda_result, 'rb') as f:
            lda_model = pk.load(f)
        return lda_model

    lda_model = model_import(facility, lda_num_topics)

    # LDA Tuning
    def get_target_docs(lda_model, topic_id):
        target_docs = [(idx, doc) for idx, doc in enumerate(lda_model.docs_by_topic[topic_id])]
        return target_docs

    def lda_tuning(target_docs):

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
        _lda_model = TopicModeling(**lda_tuning_config)
        lda_tuning_result = _lda_model.tuning()

        return lda_tuning_result

    print('LDA Tuning: {}_{}_{}'.format(facility, lda_num_topics, topic_id))
    fname_lda_tuning_result = './result/lda_tuning/news_20191223_{}_{}_{}.xlsx'.format(facility, lda_num_topics, topic_id)
    target_docs = get_target_docs(lda_model, topic_id)
    lda_tuning_result = lda_tuning(target_docs)
    save_df2excel(lda_tuning_result, fname_lda_tuning_result, verbose=True)
    # lda_tuning_result = pd.read_excel(fname_lda_tuning_result)

    # LDA Modeling
    def get_optimum_parameters(lda_tuning_result):
        max_coherence = lda_tuning_result.sort_values(by='Coherence', ascending=False).iloc[0]

        sub_lda_num_topics = int(max_coherence['Num_of_Topics'])
        sub_alpha = np.around(max_coherence['Alpha'], decimals=1)
        sub_beta = np.around(max_coherence['Beta'], decimals=1)

        return sub_lda_num_topics, sub_alpha, sub_beta

    def topic_modeling(target_docs):
        sub_lda_num_topics, sub_alpha, sub_beta = get_optimum_parameters(lda_tuning_result)
        print('LDA Modeling (Sub): {}-{}-{}'.format(sub_lda_num_topics, sub_alpha, sub_beta))

        lda_model_config = {
            'corpus': target_docs,
            'num_topics': sub_lda_num_topics,
            'alpha': sub_alpha,
            'beta': sub_beta
            }

        sub_lda_model = TopicModeling(**lda_model_config)
        sub_lda_model.learn()
        sub_lda_model.assign()
        return sub_lda_model

    sub_lda_model = topic_modeling(target_docs)
    fname_sub_lda_result = './model/lda_model_news_20191223_{}_{}_sub_{}_{}.pk'.format(facility, lda_num_topics, topic_id, sub_lda_model.num_topics)
    with open(fname_sub_lda_result, 'wb') as f:
        pk.dump(sub_lda_model, f)
    # with open(fname_sub_lda_result, 'rb') as f:
    #     sub_lda_model = pk.load(f)

    # Visualization
    import pyLDAvis
    import pyLDAvis.gensim as gensimvis

    def visualize_lda(sub_lda_model, facility, lda_num_topics, topic_id):
        visual_window = gensimvis.prepare(sub_lda_model.model, sub_lda_model.docs_for_lda, sub_lda_model.id2word)
        fname_lda_visual = './result/topic_modeling_v2/lda_visual_news_20191223_{}_{}_sub_{}.html'.format(facility, lda_num_topics, topic_id)
        pyLDAvis.save_html(visual_window, fname_lda_visual)

    visualize_lda(sub_lda_model, facility, lda_num_topics, topic_id)

for topic_id in range(lda_num_topics):
    main(facility, lda_num_topics, topic_id)