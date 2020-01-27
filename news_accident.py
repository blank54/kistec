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
import itertools
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

matplotlib.rc('font', family='NanumBarunGothic')

from kistec.object import *
from kistec.function import *
from kistec.visualize import TopicModeling, WordNetwork

kp = KistecPreprocess()
kp.build_userdic()
komoran = Komoran(userdic=kp.fname_userdic)

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

def data_import(query):
    fname_pathtofile = os.path.join(cfg.root, cfg.fdir_news_pathtofile, 'pathtofile_prep_{}_after2015.pk'.format(query))
    with open(fname_pathtofile, 'rb') as f:
        pathtofile = pk.load(f)

    corpus = []
    for fname in pathtofile:
        with open(fname, 'rb') as f:
            article = pk.load(f)
        if article.content_prep:
            corpus.append(article)

    docs = [(article.id, article.content_prep.split(' ')) for article in corpus]

    print('corpus: {}'.format(len(corpus)))
    print('docs: {}'.format(len(docs)))
    return corpus, docs

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

def wordcloud_lda(assigned_data, topic_id, fname_wordcloud):
    topic_docs = assigned_data.loc[assigned_data['topic_id'] == topic_id]

    text = []
    for idx in range(len(topic_docs)):
        text.append(topic_docs.iloc[idx]['content_prep'])

    # print(text[:3])

    wc = WordCloud(
        font_path=cfg.font_path,
        background_color='white',
        width=800,
        height=600,
        max_words=50)

    wc = wc.generate_from_text('  '.join(text))

    fig = plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(fname_wordcloud, dpi=600)
    print(fname_wordcloud)

def topic_assignment(corpus, lda_model, fname_docs_topic_assignment):
    df = defaultdict(list)
    for article in corpus:
        df['id'].append(article.id)
        df['date'].append(article.date)
        df['title'].append(article.title)
        df['category'].append(article.category)
        df['content'].append(article.content)
        df['content_prep'].append(article.content_prep)
        df['comment_list'].append(article.comment_list)
        df['comment_count'].append(article.comment_count)
        df['topic_id'].append(lda_model.tag2topic[article.id])
        df['url'].append(article.url)

    save_df2excel(pd.DataFrame(df), fname_docs_topic_assignment, verbose=True)
    print(len(df))
    return df

def timeline_graph(query, assigned_data, topic_id_list, fdir_trend_plt):
    topic_ids = '+'.join([str(id) for id in topic_id_list])
    fname_plt_frequency = os.path.join(fdir_trend_plt, '{}_{}_frequency.png'.format(query, topic_ids))
    fname_plt_comments = os.path.join(fdir_trend_plt, '{}_{}_comments.png'.format(query, topic_ids))

    date_list = sorted([datetime.strptime(str(date), '%Y%m%d') for date in assigned_data['date']])
    _from = date_list[0]
    _to = date_list[-1]
    _range = _to - _from
    x = list(set([datetime.strftime(_from + timedelta(days=d), '%Y-%m') for d in range(_range.days +1)]))

    y_freq = {time:[] for time in x}
    y_comment = {time:0 for time in x}
    topic_docs = assigned_data.loc[assigned_data['topic_id'].isin(topic_id_list)]

    for idx in range(len(topic_docs)):
        article = topic_docs.iloc[idx]
        time = datetime.strftime(datetime.strptime(str(article['date']), '%Y%m%d'), '%Y-%m')

        y_freq[time].append(article['id'])
        y_comment[time] += int(article['comment_count'])

    f1 = plt.figure(num=1, figsize=(40,30))
    plt.plot(sorted(x), [len(y) for _, y in sorted(y_freq.items(), key=lambda x:x[0])])
    plt.title('Article Frequency: {}_{}'.format(query, topic_ids))
    plt.xlabel('Month')
    plt.ylabel('Frequency')
    plt.xticks(rotation=60)
    plt.yticks(range(0, max([len(y) for _, y in y_freq.items()]), int(max([len(y) for _, y in y_freq.items()])/10)))
    plt.savefig(fname_plt_frequency, dpi=600)
    plt.clf()

    f2 = plt.figure(num=1, figsize=(40,30))
    plt.plot(sorted(x), [y for _, y in sorted(y_comment.items(), key=lambda x:x[0])])
    plt.title('Comments Count: {}_{}'.format(query, topic_ids))
    plt.xlabel('Month')
    plt.ylabel('Comments')
    plt.xticks(rotation=60)
    plt.yticks(range(0, max([y for _, y in y_comment.items()]), int(max([y for _, y in y_comment.items()])/10)))
    plt.savefig(fname_plt_comments, dpi=600)
    plt.clf()

    print(fname_plt_frequency)
    print(fname_plt_comments)

def main():
    query_list = ['교량+사고', '터널+사고', '건물+사고']
    for query in query_list:
    # query = '교량+사고'
        fname_lda_tuning_result = os.path.join(cfg.root, cfg.fdir_lda_tuning_news_accident, '{}.xlsx'.format(query))
        fname_lda_model = os.path.join(cfg.root, cfg.fdir_lda_model_news, '{}.pk'.format(query))
        fname_lda_visual = os.path.join(cfg.root, cfg.fdir_lda_visual_news, '{}.html'.format(query))
        fname_docs_topic_assignment = os.path.join(cfg.root, cfg.fdir_topic_assignment_news, '{}.xlsx'.format(query))
        fdir_trend_plt = os.path.join(cfg.root, cfg.fdir_trend_plt_news)

        corpus, docs = data_import(query)

        # print('LDA Tuning: ({})'.format(query))
        # lda_tuning(docs, fname_lda_tuning_result)

        # lda_parameters = get_optimum_parameters(fname_lda_tuning_result)
        # print('LDA Modeling: {} / num_topics: {} / alpha: {} / beta: {}'.format(query, lda_parameters[0], lda_parameters[1], lda_parameters[2]))
        # lda_model = topic_modeling(docs, lda_parameters, fname_lda_model)
        with open(fname_lda_model, 'rb') as f:
            lda_model = pk.load(f)

        # print('Save LDA Visualization: {}'.format(query))
        # visualize_lda(lda_model, fname_lda_visual)

        # print('Topic Assignment: {}'.format(query))
        # topic_assignment(corpus, lda_model, fname_docs_topic_assignment)
        assigned_data = pd.read_excel(fname_docs_topic_assignment)

        # print('WordCloud: ({})'.format(query))
        # for topic_id in range(lda_model.num_topics):
        #     fname_wordcloud = os.path.join(cfg.root, cfg.fdir_word_cloud, 'news/{}_{}.png'.format(query, topic_id))
        #     wordcloud_lda(assigned_data, topic_id, fname_wordcloud)

        print('Line Graph: {}'.format(query))
        for topic_id in range(lda_model.num_topics):
            topic_id_list = [topic_id]
            timeline_graph(query, assigned_data, topic_id_list, fdir_trend_plt)
main()