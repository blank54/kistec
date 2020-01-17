#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import sys
sys.path.append("/data/blank54/workspace/kistec/src/")
from visualize import TopicModeling

import re
import pickle as pk
import pyLDAvis
import pyLDAvis.gensim

# Data Import
def data_import(fname_tweet):
    fname_pos_target = "{}_pos_target.pk".format(fname_tweet)
    with open(fname_pos_target, "rb") as f:
        data = pk.load(f)

    docs = [(key, text) for key, text in data.items()]
    docs = [(key, [w for w in re.sub("[0-9ㄱ-ㅎ]", "", " ".join(text)).split(" ") if w]) for key, text in docs]
    return docs

fname_tweet = "tweet_20191024"
docs = data_import(fname_tweet)

# LDA Modeling
def topic_modeling(docs, fname_tweet, operate=True, num_topics=10):
    fname_lda_model = "{}_lda_model_{}.pk".format(fname_tweet, num_topics)
    if operate:
        lda_model = TopicModeling(docs)
        lda_model.num_topics = num_topics
        lda_model.learn()

        with open(fname_lda_model, "wb") as f:
            pk.dump(lda_model, f)

    else:
        with open(fname_lda_model, "rb") as f:
            lda_model = pk.load(f)

    return lda_model

lda_model = topic_modeling(docs, fname_tweet, operate=True)

# # Visualize LDA Model (Jupyter Notebook Only)
# def visualize_in_jupyter(lda_model):
#     pyLDAvis.enable_notebook()
#     return pyLDAvis.gensim.prepare(lda_model.model, lda_model.docs_for_lda, lda_model.id2word)

# visual_window = visualize_in_jupyter(lda_model)