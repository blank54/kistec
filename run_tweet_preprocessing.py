# Configuration
import sys
sys.path.append("/data/blank54/workspace/kistec/src/")
from data_import import SaveExcel
from embedding import doc2vec, TFIDF

import os
import re
import pickle as pk
import pandas as pd
import itertools
from collections import defaultdict
from sklearn.decomposition import PCA

from konlpy.tag import Kkma
kkma = Kkma()

# Data Import
def data_import(fname_tweet):
    fdir = "/data/blank54/workspace/kistec_workspace/08_tweet/"
    fname = "{}.xlsx".format(fname_tweet)
    
    tweet_data = pd.read_excel(os.path.join(fdir, fname))
    keys = list(tweet_data["key"])
    docs = {}
    for i in range(len(tweet_data)):
        target = tweet_data.iloc[i]
        key = target["key"]
        text = target["content"]
        docs[key] = text
    print("Data Imported: {}".format(fname))
    return keys, docs

fname_tweet = "tweet_20191024"
keys, docs = data_import(fname_tweet)

# Data Cleaning
def data_clean(docs):
    docs_clean = {}
    for idx, key in enumerate(docs):
        # Eliminate duplicated characters
        doc = "".join(ch for ch, _ in itertools.groupby(str(docs[key])))

        # Eliminate short data
        if len(doc.strip().split(" ")) >= 5:
            docs_clean[key] = doc
        else:
            continue
    return docs_clean

docs_clean = data_clean(docs)

# PoS Tagging
def pos_tagging(docs, fname_tweet, operate=True):
    fname_pos = "{}_pos.pk".format(fname_tweet)

    if operate:
        docs_pos = {}
        for idx, key in enumerate(docs):
            sys.stdout.write("\rPoS Tagging: {}/{}".format(idx+1, len(docs)))
            try:
                docs_pos[key] = kkma.pos(docs[key])
            except:
                docs_pos_error.append((key, docs[key]))

        with open(fname_pos, "wb") as f:
            pk.dump(docs_pos, f)
        print("\nSaved PoS Result: {}".format(fname_pos))

    else:
        with open(fname_pos, "rb") as f:
            docs_pos = pk.load(f)
        print("\nLoaded PoS Result: {}".format(fname_pos))

    return docs_pos

docs_pos = pos_tagging(docs_clean, fname_tweet, operate=False)

# Extract Target PoS
def extract_target_pos(docs_pos, fname_tweet, operate=True):
    fname_pos_target = "{}_pos_target.pk".format(fname_tweet)

    if operate:
        target_pos = ["N"]
        docs_pos_target = {}
        docs_pos_error = []

        for idx, key in enumerate(docs_pos):
            sys.stdout.write("\rPoS Tagging: {}/{}".format(idx+1, len(docs_pos)))
            docs_pos_target[key] = [token for token, tag in docs_pos[key] if tag[0] in target_pos]

        with open(fname_pos_target, "wb") as f:
            pk.dump(docs_pos_target, f)
        print("\nSaved Target PoS Result: {}".format(fname_pos_target))

        # Save as Excel
        fname_pos_target_excel = re.sub(".pk", ".xlsx", fname_pos_target)
        _keys = list(docs_pos_target.keys())    
        _docs_original = [docs[key] for key in docs_pos_target]
        _docs_pos_target = [doc for key, doc in docs_pos_target.items()]

        df_data = {}
        df_data["key"] = _keys
        df_data["content"] = _docs_original
        df_data["content_pos_target"] = _docs_pos_target
        df_docs_pos_target = pd.DataFrame(df_data)
        SaveExcel().save(df_docs_pos_target, fname_pos_target_excel)

    else:
        with open(fname_pos_target, "rb") as f:
            docs_pos_target = pk.load(f)
        print("\nLoaded Target PoS Result: {}".format(fname_pos_target))

    return docs_pos_target

docs_pos_target = extract_target_pos(docs_pos, fname_tweet, operate=True)

# Doc2Vec Embedding
def doc2vec_embedding(docs, fname_tweet, operate=True):
    fname_d2v = "{}_d2v.pk".format(fname_tweet)

    if operate:
        docs_for_d2v = [(key, " ".join(docs[key])) for key in docs]
        d2v_model = doc2vec(docs_for_d2v)

        with open(fname_d2v, "wb") as f:
            pk.dump(d2v_model, f)
        print("Saved Doc2Vec Model: {}".format(fname_d2v))

    else:
        with open(fname_d2v, "rb") as f:
            d2v_model = pk.load(f)
        print("Loaded Doc2Vec Model: {}".format(fname_d2v))

    return d2v_model

d2v_model = doc2vec_embedding(docs_pos_target, fname_tweet, operate=False)
d2v_vector = d2v_model.docvecs

# TFIDF Embedding
def tfidf_embedding(docs, fname_tweet, operate=True):
    fname_tfidf = "{}_tfidf.pk".format(fname_tweet)
    if operate:
        docs_for_tfidf = [(key, doc) for key, doc in docs.items()]
        tfidf_model = TFIDF(docs_for_tfidf)
        tfidf_model.train()

        with open(fname_tfidf, "wb") as f:
            pk.dump(tfidf_model, f)
        print("Saved TFIDF Matrix: {}".format(fname_tfidf))

    else:
        with open(fname_tfidf, "rb") as f:
            tfidf_model = pk.load(f)
        print("Loaded TFIDF Matrix: {}".format(fname_tfidf))

    return tfidf_model

tfidf_model = tfidf_embedding(docs_pos_target, fname_tweet, operate=False)
tfidf_vector = tfidf_model.tfidf_matrix

# PCA Dimension Reduction
def tfidf2pca(tfidf_vector, vector_size=100, operate=True):
    fname_tfidf_pos = "{}_tfidf_pos.pk".format(fname_tweet)

    if operate:
        pca_model = PCA(n_components=vector_size)
        pca_vector = pca_model.fit_transform(tfidf_vector)

        with open(fname_tfidf_pos, "wb") as f:
            pk.dump(pca_vector, f)

    else:
        with open(fname_tfidf_pos, "rb") as f:
            pca_vector = pk.load(f)

    return pca_vector

pca_vector = tfidf2pca(tfidf_vector, operate=False)

# Topic Modeling
