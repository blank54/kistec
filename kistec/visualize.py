#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import itertools
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

matplotlib.rc('font', family='NanumBarunGothic')

class PPMI:
    def __init__(self, docs):
        self.docs = docs
        self.words = sorted(list(set(itertools.chain(*docs))))
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.id2word = {i: w for w, i in self.word2id.items()}

        self.verbose = True
        self.ppmi_result = ''

    def compute(self):
        if len(self.ppmi_result) == 0:
            word_cnt = len(self.words)
            word_freq = Counter(list(itertools.chain(*self.docs)))
            
            total_prob = {self.word2id[word]: cnt/word_cnt for word, cnt in word_freq.items()}
            
            joint_cnt = np.zeros((word_cnt, word_cnt))
            for idx, doc in enumerate(self.docs):
                if self.verbose:
                    sys.stdout.write('\rCalculating Joint Probability {}/{}'.format(idx+1, len(self.docs)))
                for comb in list(itertools.combinations(doc, 2)):
                    w1, w2 = comb
                    joint_cnt[self.word2id[w1], self.word2id[w2]] += 1
            print()
            joint_prob = joint_cnt/word_cnt

            def regularize(value):
                if value == 0:
                    return 0.00001
                else:
                    return value

            def positive(value):
                if value > 0:
                    return value
                else:
                    return 0

            def ppmi_value(word1, word2):
                w1_id = self.word2id[word1]
                w2_id = self.word2id[word2]
                
                numerator = regularize(joint_prob[w1_id, w2_id])
                denominator = regularize(total_prob[w1_id] * total_prob[w2_id])
                return positive(np.round(np.log(numerator / denominator), 3))

            ppmi_result = []
            for comb in list(itertools.combinations(self.words, 2)):
                word1, word2 = comb
                if self.verbose:
                    sys.stdout.write('\rComputing PPMI [{:>5s}] x [{:>5s}]'.format(word1, word2))
                
                ppmi_result.append([word1, word2, ppmi_value(word1, word2)])
            self.ppmi_result = pd.DataFrame(sorted(ppmi_result, key=lambda x:x[2], reverse=True), columns=['word1', 'word2', 'ppmi_value'])

    def occur_with(self, query):
        if len(self.ppmi_result) == 0:
            self.compute()

        target_ppmi = {}
        query_index = [any((if1, if2)) for if1, if2 in zip(self.ppmi_result['word1'] == query, self.ppmi_result['word2'] == query)]
        for idx, is_word1 in enumerate(list(self.ppmi_result[query_index]['word1'] != query)):
            if is_word1:
                with_word = self.ppmi_result[query_index].iloc[idx]['word1']
            else:
                with_word = self.ppmi_result[query_index].iloc[idx]['word2']
            ppmi_value = self.ppmi_result[query_index].iloc[idx]['ppmi_value']
            
            target_ppmi[with_word] = ppmi_value
        return pd.DataFrame([(w, v) for w, v in target_ppmi.items()], columns=['word', 'ppmi_value'])

class WordNetwork:
    def __init__(self, **kwargs):
        self.docs = kwargs.get('docs', '') # list of list of words: [[w1, w2, ...], [w3, ...], ...]
        self.count_option = kwargs.get('count_option', 'dist')
        self.top_n = kwargs.get('top_n', 100)

        self.fname_combs = kwargs.get('fname_combs', './word_combs_{}_{}.pk'.format(self.count_option, self.top_n))
        self.calculate_combs = kwargs.get('calculate_combs', True)
        self._combinations = defaultdict(float)

        self.save_plt = kwargs.get('save_plt', True)
        self.fname_plt = kwargs.get('fname_plt', './word_network_{}_{}.png'.format(self.count_option, self.top_n))
        self.show_plt = kwargs.get('show_plt', False)

    def combinations(self):
        if not self._combinations:
            if self.calculate_combs:
                combs = defaultdict(float)
                print('Calculating Word Combinations ...')
                with tqdm(total=len(self.docs)) as pbar:
                    for idx, doc in enumerate(self.docs):
                        doc_len = len(doc)
                        for i in range(doc_len):
                            for j in range(i+1, doc_len):
                                w1 = doc[i]
                                w2 = doc[j]
                                key = '__'.join((w1, w2))

                                if self.count_option == 'occur':
                                    combs[key] += 1
                                elif self.count_option == 'dist':
                                    dist = np.abs(j-i)/(doc_len-1)
                                    combs[key] += dist
                        pbar.update(1)

                self._combinations = combs
                with open(self.fname_combs, 'wb') as f:
                    pk.dump(combs, f)

            else:
                with open(self.fname_combs, 'rb') as f:
                    self._combinations = pk.load(f)
        return self._combinations

    def top_n_combs(self):
        sorted_combs = sorted(self.combinations().items(), key=lambda x:x[1], reverse=True)
        return {key: np.round(value, 3) for key, value in sorted_combs[:self.top_n]}

    def network(self):
        combs = self.top_n_combs()
        combs_df = pd.DataFrame(combs.items(), columns=['comb', 'count'])
        # combs_df['comb'] : '도로__위험', '도로__사고', ...
        # combs_df['count'] : 47.861, 25.977, ...

        d = combs_df[:-10].set_index('comb').T.to_dict('record')
        G = nx.Graph()
        for k, v in d[0].items():
            w1, w2 = k.split('__')
            G.add_edge(w1, w2, weight=(v*10))

        fig, ax = plt.subplots(figsize=(10,8), dpi=600) # 그림 사이즈 & 해상도
        pos = nx.spring_layout(G, k=1)

        nx.draw_networkx(G, pos,
                         node_size=25,
                         font_size=0,
                         width=1,
                         edge_color='grey',
                         node_color='purple',
                         with_labels=True,
                         ax=ax)

        for key, value in pos.items():
            x, y = value[0], value[1] + 0.025
            ax.text(x, y, s=key,
                    bbox=dict(facecolor='white', alpha=0, edgecolor='white'), # 글씨 배경 색깔 & 투명도
                    horizontalalignment='center', fontsize=12) # 글씨 크기는 이걸 바꾸면됨
        
        if self.save_plt:
            plt.savefig(self.fname_plt, dpi=600)
            print('Saved: Word Network ({})'.format(self.fname_plt))
        
        if self.show_plt:
            plt.show()

class TopicModeling:
    '''
    Refer to: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    '''

    def __init__(self, **kwargs):
        self.corpus = kwargs.get('corpus', '')
        self.docs = [doc for _, doc in self.corpus]
        self.tags = [tag for tag, _ in self.corpus]
        self.id2word = corpora.Dictionary(self.docs)
        self.docs_for_lda = [self.id2word.doc2bow(doc) for doc in self.docs]

        self.num_topics = kwargs.get('num_topics', 10)
        self.alpha = kwargs.get('alpha', 'auto')
        self.beta = kwargs.get('beta', 'auto')
        self.model = None

        self._coherence = None
        self.tuning_result = None

        self.tag2topic = {}
        self.docs_by_topic = defaultdict(list)

        self.min_topics = kwargs.get('min_topics', 3)
        self.max_topics = kwargs.get('max_topics', 10)
        self.topics_step = kwargs.get('topics_step', 1)
        self.min_alpha = kwargs.get('min_alpha', 0.05)
        self.max_alpha = kwargs.get('max_alpha', 1)
        self.alpha_step = kwargs.get('alpha_step', 0.3)
        self.alpha_symmetric = kwargs.get('alpha_symmetric', True)
        self.alpha_asymmetric = kwargs.get('alpha_asymmetric', True)
        self.min_beta = kwargs.get('min_beta', 0.05)
        self.max_beta = kwargs.get('max_beta', 1)
        self.beta_step = kwargs.get('beta_step', 0.3)
        self.beta_symmetric = kwargs.get('beta_symmetric', True)

    def learn(self):
        if not self.model:
            lda_model = LdaModel(
                corpus=self.docs_for_lda,
                id2word=self.id2word,
                num_topics=self.num_topics,
                random_state=100,
                update_every=1,
                chunksize=100,
                passes=10,
                alpha=self.alpha,
                eta=self.beta,
                per_word_topics=True)
            self.model = lda_model

    def coherence(self):
        if not self._coherence:
            coherence_model = CoherenceModel(model=self.model,
                                             texts=self.docs,
                                             dictionary=self.id2word,
                                             coherence='c_v')
            self._coherence = coherence_model.get_coherence()

        return self._coherence

    def assign(self):
        if len(self.docs_by_topic) == 0:
            result = self.model[self.docs_for_lda]
            with tqdm(total=len(self.tags)) as pbar:
                for idx, tag in enumerate(self.tags):
                    row = result[idx]
                    topic_id = sorted(row[0], key=lambda x:x[1], reverse=True)[0][0]

                    self.tag2topic[tag] = topic_id
                    self.docs_by_topic[topic_id].append(self.docs[idx])
                    pbar.update(1)
        return self.docs_by_topic

    def tuning(self):
        topics_range = range(self.min_topics, self.max_topics+1, self.topics_step)

        alpha = list(np.arange(self.min_alpha, self.max_alpha+0.000000000001, self.alpha_step))
        if self.alpha_symmetric:
            alpha.append('symmetric')
        if self.alpha_asymmetric:
            alpha.append('asymmetric')

        beta = list(np.arange(self.min_beta, self.max_beta+0.000000000001, self.beta_step))
        if self.beta_symmetric:
            beta.append('symmetric')

        total_progress = len(topics_range) * len(alpha) * len(beta)
        tuning_result = defaultdict(list)
        with tqdm(total=total_progress) as pbar:
            for k in topics_range:
                for a in alpha:
                    for b in beta:
                        _model_config = {
                            'corpus': self.corpus,
                            'num_topics': k,
                            'alpha': a,
                            'beta': b
                            }
                        _model = TopicModeling(**_model_config)
                        _model.learn()

                        tuning_result['Num_of_Topics'].append(k)
                        tuning_result['Alpha'].append(a)
                        tuning_result['Beta'].append(b)
                        tuning_result['Coherence'].append(_model.coherence())
                        pbar.update(1)
        self.tuning_result = pd.DataFrame(tuning_result)
        return self.tuning_result