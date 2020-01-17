#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kistec.preprocess import KistecPreprocess

import sys
import itertools
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

kpr = KistecPreprocess()

class TFIDF():
    def __init__(self, docs):
        self.docs = docs
        self.doc2id = {}
        self.words = sorted(list(set(itertools.chain(*[doc for _, doc in docs]))))
        self.word2id = {w: i for i, w in enumerate(self.words)}

        self.balance = 0.01
        self.tfidf_matrix = ""

    def tf(self):
        tf_shape = np.zeros((len(self.docs), len(self.words)))
        term_frequency = self.balance + ((1-self.balance) * tf_shape.astype(float))
        for doc_id, (tag, doc) in enumerate(self.docs):
            self.doc2id[tag] = doc_id
            word_freq = Counter(doc)
            for word in set(doc):
                word_id = self.word2id[word]
                term_frequency[doc_id, word_id] = word_freq[word]
        return term_frequency

    def df(self):
        document_frequency = np.array([len([True for doc in self.docs if word in doc]) for word in self.words])
        return document_frequency
    
    def idf(self):
        inverse_document_frequency = np.log(len(self.docs) / (1+self.df()))
        return inverse_document_frequency

    def train(self, vector_size=100):
        if len(self.tfidf_matrix) == 0:
            tfidf_matrix = self.tf() * self.idf()
            self.tfidf_matrix = tfidf_matrix

    def most_similar(self, tag, top_n=10):
        if len(self.tfidf_matrix) == 0:
            print("Error: train TFIDF model first")
            return None

        id2doc = {i:tag for tag, i in self.doc2id.items()}
        target_doc_id = self.doc2id[tag]
        target_doc_vector = self.tfidf_matrix[target_doc_id,]
        similarity_score = []
        for i in range(len(self.docs)):
            tag = id2doc[i]
            refer_doc_vector = self.tfidf_matrix[i,]
            score = cosine_similarity([target_doc_vector], [refer_doc_vector])
            similarity_score.append((tag, score))

        return sorted(similarity_score, key=lambda pair:pair[1], reverse=True)[:top_n]

class KistecWord2Vec:
    '''
    Input docs: [[word, word, ...], [word, word, ...], ...]
    '''

    def __init__(self, **kwargs):
        self.docs = kwargs.get('docs', '')
        self.docs_for_w2v = [kpr.tokenize(doc, do_thes=False) for doc in self.docs]
        self.parameters = kwargs.get('parameters', {})

        self.model = None
        self.save = kwargs.get('save', True)

    def learn(self):
        model = Word2Vec(size=self.parameters.get('size', 100),
                         window=self.parameters.get('window', 5),
                         min_count=self.parameters.get('min_count', 1),
                         workers=self.parameters.get('workers', 4),
                         sg=self.parameters.get('architecture', 1),
                         alpha=self.parameters.get('alpha', 0.025),
                         min_alpha=self.parameters.get('min_alpha', 0.00025))
        model.build_vocab(self.docs_for_w2v)
        
        max_epoch = self.parameters.get('max_epoch', 100)
        with tqdm(total=max_epoch) as pbar:
            for epoch in range(max_epoch):
                model.train(
                    sentences=self.docs_for_w2v,
                    total_examples=model.corpus_count,
                    epochs=epoch)
                model.alpha -= 0.0002
                pbar.update(1)

        self.model = model
        print('Word2Vec Learning: Done!!!')

        if self.save:
            with open('./model/word2vec_model.pk', 'wb') as f:
                pk.dump(model, f)
            print('Word2Vec Model Saved: \'./word2vec_model.pk\'')

class KistecDoc2Vec:
    '''
    Input docs: [(tag, doc), (tag, doc), ...]
    '''

    def __init__(self, **kwargs):
        self.docs = kwargs.get('docs', '')
        self.docs_for_d2v = [TaggedDocument(words=doc, tags=[tag]) for tag, doc in self.docs]
        self.parameters = kwargs.get('parameters', {})
        
        self.model = None
        self.save = kwargs.get('save', True)

    def learn(self):
        model = Doc2Vec(vector_size=self.parameters.get('vector_size', 100),
                        alpha=self.parameters.get('alpha', 0.025),
                        min_alpha=self.parameters.get('min_alpha', 0.00025),
                        min_count=self.parameters.get('min_count', 5),
                        window=self.parameters.get('window', 100),
                        workers=self.parameters.get('workers', 4),
                        dm=self.parameters.get('dm', 1))
        model.build_vocab(self.docs_for_d2v)

        max_epoch = self.parameters.get('max_epoch', 100)
        with tqdm(total=max_epoch) as pbar:
            for epoch in range(max_epoch):
                model.train(
                    documents=self.docs_for_d2v,
                    total_examples=model.corpus_count,
                    epochs=epoch)
                model.alpha -= 0.0002
                pbar.update(1)

        self.model = model
        print('Doc2Vec Learning: Done!!!')

        if self.save:
            with open('./doc2vec_model.pk', 'wb') as f:
                pk.dump(model, f)
            print('Doc2Vec Model Saved: \'./doc2vec_model.pk\'')