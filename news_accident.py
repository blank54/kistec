#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
from config import Config
import pickle as pk
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from konlpy.tag import Komoran
from datetime import datetime, timedelta
import pyLDAvis
import pyLDAvis.gensim as gensimvis
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

matplotlib.rc('font', family='NanumBarunGothic')

from kistec.data_import import *
from kistec.preprocess import KistecPreprocess
from kistec.visualize import TopicModeling, WordNetwork

kp = KistecPreprocess()
kp.build_userdic()
komoran = Komoran(userdic=kp.fname_userdic)

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

def data_import(facility):
    fname_corpus_news_preprocessed = os.path.join(cfg.root, cfg.fname_corpus_news_preprocessed)
    with open(fname_corpus_news_preprocessed, 'rb') as f:
        corpus = pk.load(f)

    target_corpus = [article for article in corpus[facility] if '사고' in article.content]    
    target_docs = [(article.id, article.content_prep) for article in corpus[facility] if '사고' in article.content]

    print(len(corpus[facility]))
    print(len(target_docs))
    return target_corpus, target_docs

def lda_tuning(docs, fname_lda_tuning_result):
    lda_tuning_config = {
        'corpus': docs,
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
    save_df2excel(lda_tuning_result, fname_lda_tuning_result, verbose=True)
    print('done!!!')

def get_optimum_parameters(fname_lda_tuning_result):
    lda_tuning_result = pd.read_excel(fname_lda_tuning_result)
    max_coherence = lda_tuning_result.sort_values(by='Coherence', ascending=False).iloc[0]

    lda_num_topics = int(max_coherence['Num_of_Topics'])
    alpha = np.around(max_coherence['Alpha'], decimals=1)
    beta = np.around(max_coherence['Beta'], decimals=1)

    return (lda_num_topics, alpha, beta)

def topic_modeling(docs, lda_parameters, fname_lda_model):
    lda_num_topics, alpha, beta = lda_parameters

    lda_model_config = {
        'corpus': docs,
        'num_topics': lda_num_topics,
        'alpha': alpha,
        'beta': beta
        }

    lda_model = TopicModeling(**lda_model_config)
    lda_model.learn()
    lda_model.assign()

    with open(fname_lda_model, 'wb') as f:
        pk.dump(lda_model, f)
    return lda_model

def visualize_lda(lda_model, fname_lda_visual):
    visual_window = gensimvis.prepare(lda_model.model, lda_model.docs_for_lda, lda_model.id2word)
    pyLDAvis.save_html(visual_window, fname_lda_visual)
    print(fname_lda_visual)

def topic_assignment(corpus, lda_model, fname_docs_topic_assignment):
    articles = []
    for article in corpus:
        article.topic_id = lda_model.tag2topic[article.id]
        articles.append(article)

    df = articles2df(articles)
    save_df2excel(df, fname_docs_topic_assignment, verbose=True)
    print(len(df))
    return df

def timeline_graph(facility, assigned_data, topic_id, fdir_trend_plt):
    fname_plt_frequency = os.path.join(fdir_trend_plt, '{}+사고_{}_frequency.png'.format(facility, topic_id))
    fname_plt_comments = os.path.join(fdir_trend_plt, '{}+사고_{}_comments.png'.format(facility, topic_id))

    date_list = sorted([datetime.strptime(str(date), '%Y%m%d') for date in assigned_data['date']])
    _from = date_list[0]
    _to = date_list[-1]
    _range = _to - _from
    x = list(set([datetime.strftime(_from + timedelta(days=d), '%Y-%m') for d in range(_range.days +1)]))

    y_freq = {time:[] for time in x}
    y_comment = {time:0 for time in x}
    topic_docs = assigned_data.loc[assigned_data['topic_id'] == topic_id]
    for idx in range(len(topic_docs)):
        article = topic_docs.iloc[idx]
        time = datetime.strftime(datetime.strptime(str(article['date']), '%Y%m%d'), '%Y-%m')

        y_freq[time].append(article['id'])
        y_comment[time] += int(article['comment_count'])

    f1 = plt.figure(num=1, figsize=(40,30))
    plt.plot(x, [len(articles) for time, articles in sorted(y_freq.items())])
    plt.title('Article Frequency: {}+사고_{}'.format(facility, topic_id))
    plt.xlabel('Month')
    plt.ylabel('Frequency')
    plt.xticks(rotation=60)
    plt.savefig(fname_plt_frequency, dpi=600)

    f2 = plt.figure(num=1, figsize=(40,30))
    plt.plot(x, [y for time, y in sorted(y_comment.items())])
    plt.title('Comments Count: {}+사고_{}'.format(facility, topic_id))
    plt.xlabel('Month')
    plt.ylabel('Comments')
    plt.xticks(rotation=60)
    plt.savefig(fname_plt_comments, dpi=600)

    print(fname_plt_frequency)
    print(fname_plt_comments)

def main():
    facility_list = ['교량', '터널', '건물']
    for facility in facility_list:
        fname_lda_tuning_result = os.path.join(cfg.root, cfg.fdir_lda_tuning_news_accident, '{}.xlsx'.format(facility))
        fname_lda_model = os.path.join(cfg.root, cfg.fdir_lda_model_news, '{}.pk'.format(facility))
        fname_lda_visual = os.path.join(cfg.root, cfg.fdir_lda_visual_news, '{}+사고.html'.format(facility))
        fname_docs_topic_assignment = os.path.join(cfg.root, cfg.fdir_topic_assignment, '{}+사고.xlsx'.format(facility))
        fdir_trend_plt = os.path.join(cfg.root, cfg.fdir_trend_plt_news)


        target_corpus, target_docs = data_import(facility)
        # print('LDA Tuning: {}'.format(facility))
        # lda_tuning(facility, target_docs, fname_lda_tuning_result)

        # lda_parameters = get_optimum_parameters(fname_lda_tuning_result)
        # print('LDA Modeling: {} / num_topics: {} / alpha: {} / beta: {}'.format(facility, lda_parameters[0], lda_parameters[1], lda_parameters[2]))
        # lda_model = topic_modeling(target_docs, lda_parameters, fname_lda_model)
        with open(fname_lda_model, 'rb') as f:
            lda_model = pk.load(f)

        # print('Save LDA Visualization: {}'.format(facility))
        # visualize_lda(lda_model, fname_lda_visual)

        # print('Topic Assignment: {}'.format(facility))
        # topic_assignment(target_corpus, lda_model, fname_docs_topic_assignment)

        print('Line Graph: {}'.format(facility))
        assigned_data = pd.read_excel(fname_docs_topic_assignment)
        for topic_id in range(lda_model.num_topics):
	        timeline_graph(facility, assigned_data, topic_id, fdir_trend_plt)
main()