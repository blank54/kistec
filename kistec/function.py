#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import pickle as pk
import pandas as pd
from config import Config
from tqdm import tqdm
from collections import defaultdict

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

def make_dir(fname):
    if fname.split('/')[:-1]:
        fdir = '/'.join(fname.split('/')[:-1])
        os.makedirs(fdir, exist_ok=True)

def save_df2excel(data, fname, verbose=False):
    make_dir(fname)
        
    writer = pd.ExcelWriter(fname)
    data.to_excel(writer, "Sheet1", index=False)
    writer.save()
    if verbose:
        print("Saved data as: {}".format(fname))

def articles2df(articles):
    data = defaultdict(list)
    for article in articles:
        data['id'].append(article.id)
        data['date'].append(article.date)
        data['content'].append(article.content)
        data['content_nouns'].append(article.content_prep)
        data['comment_list'].append(article.comment_list)
        data['comment_count'].append(article.comment_count)
        data['url'].append(article.url)
        data['topic_id'].append(article.topic_id)
    return pd.DataFrame(data)

class KistecPreprocess:
    def _manage_duplicate(self):
        with open(self.fname_userword, 'r', encoding='utf-8') as f:
            userword_set = sorted(set(f.read().strip().split('\n')))
        with open(self.fname_userword, 'w', encoding='utf-8') as f:
            f.write('\n'.join(userword_set))

        with open(self.fname_stop_list, 'r', encoding='utf-8') as f:
            stop_list_set = sorted(set(f.read().strip().split('\n')))
        with open(self.fname_stop_list, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stop_list_set))

        with open(self.fname_synonyms, 'r', encoding='utf-8') as f:
            synonyms_set = sorted(set(f.read().strip().split('\n')))
        with open(self.fname_synonyms, 'w', encoding='utf-8') as f:
            f.write('\n'.join(synonyms_set))

    def __init__(self, **kwargs):
        self.do_synonym = kwargs.get('do_synonym', True)
        self.fname_userword = kwargs.get('fname_userword', os.path.join(cfg.root, cfg.fname_userword))
        self.fname_userdic = kwargs.get('fname_userdic', os.path.join(cfg.root, cfg.fname_userdic))
        self.fname_stop_list = kwargs.get('fname_stop_list', os.path.join(cfg.root, cfg.fname_stop_list))
        self.fname_synonyms = kwargs.get('fname_synonyms', os.path.join(cfg.root, cfg.fname_synonyms))

        self._manage_duplicate()
        self.stop_list = self._stop_list()

    def _stop_list(self):
        with open(self.fname_stop_list, 'r', encoding='utf-8') as f:
            return sorted(set(f.read().strip().split('\n')), reverse=False)

    def build_userdic(self):
        with open(self.fname_userword, 'r', encoding='utf-8') as f:
            wordlist = f.read().strip().split('\n')
            if not wordlist:
                return None
            userdic = '\n'.join([str(w+'\tNNP') for w in wordlist if len(w)>0])
        with open(self.fname_userdic, 'w', encoding='utf-8') as f:
            f.write(re.sub('\ufeff', '', userdic))

    def synonym(self, text):
        with open(self.fname_synonyms, 'r', encoding='utf-8') as f:
            data = f.read()
            if not data:
                return text

            synonyms = re.sub('\ufeff', '', data.strip())
            pairs = [tuple(pair.split('  ')) for pair in synonyms.split('\n')]

        if any([True for l, r in pairs if l in text]):
            for l, r in pairs:
                text = re.sub(l, r, text)
        return text

    def tokenize(self, text):
        if self.do_synonym:
            text = self.synonym(text)

        # cleaned_text = re.sub('[^ a-zA-Zㄱ-ㅣ가-힣]+', ' ', text).strip()
        cleaned_text = text
        tokenized_text = [w.strip() for w in re.split(' |  |\n', cleaned_text) if len(w)>0]
        return tokenized_text

    def stopword_removal(self, text, return_type='list'):
        result = [w for w in self.tokenize(text) if w not in self.stop_list]
        if return_type == 'list':
            return result
        elif return_type == 'str':
            return ' '.join(result)

KistecPreprocess()._manage_duplicate()