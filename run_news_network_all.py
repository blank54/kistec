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

fname_corpus = './data/corpus_news_20191223.pk'
with open(fname_corpus, 'rb') as f:
    corpus = pk.load(f)

# Preprocess
def preprocess(fdir):
    preprocess_config = {
        # 'fname_userword': './thesaurus/userword.txt',
        # 'fname_userdic': './thesaurus/userdic.txt',
        'fname_stop': './thesaurus/stop_list.txt',
        # 'fname_synonyms': './thesaurus/synonyms.txt'
        }
    kp = KistecPreprocess(**preprocess_config)

    result = defaultdict(list)
    for facility in corpus.keys():
        print('Preprocessing: {}'.format(facility))
        docs = corpus[facility]
        with tqdm(total=len(corpus[facility])) as pbar:
            for idx, doc in enumerate(docs):
                result[facility].append((idx, kp.stopword_removal(doc.content)))
                pbar.update(1)
    return result

fname_prep = './data/preprocessed/news_20191223_v1.pk'

corpus_prep = preprocess('./articles_20191223')
with open(fname_prep, 'wb') as f:
    pk.dump(corpus_prep, f)
with open(fname_prep, 'rb') as f:
    corpus_prep = pk.load(f)

# TODO:
# def get_lda_parameters(facility):
#     with open('./data/lda_parameters.txt', 'r', encoding='utf-8') as f:
#         parameters = f.read()
#     return

# LDA Modeling
def topic_modeling(corpus, facility, parameters):
    # TODO: parameters
    # lda_model_config = {
    #     'corpus': corpus[facility]
    #     }

    lda_model = TopicModeling(**lda_model_config)
    lda_model.learn()
    lda_model.assign()

    
    return lda_model

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
                           'fname_plt': './result/word_network_v2/word_network_news_20191223_{}_{}_{}'.format(facility, topic_id, count_option, top_n)
                           }
    word_network = WordNetwork(**word_network_config)
    word_network.network()

# main
for facility in corpus_prep:
    # parameters = get_lda_parameters(facility)
    lda_model = topic_modeling(corpus=corpus_prep, facility=facility, parameters=parameters)

    fname_lda_result = './result/topic_modeling/lda_model_news_20191223_{}_{}.pk'.format(facility, lda_num_topics)
    with open(fname_lda_result, 'wb') as f:
        pk.dump(lda_model, f)
    with open(fname_lda_result, 'rb') as f:
        lda_model = pk.load(f)

    for topic_id in range(lda_model.num_topics):
        count_option = 'dist'
        top_n = 100
        draw_work_network(lda_model, topic_id, count_option, top_n)