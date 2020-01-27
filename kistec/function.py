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

def makedir(path):
    if path.endswith('/'):
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

def save_df2excel(data, fname, verbose=False):
    makedir(fname)
        
    writer = pd.ExcelWriter(fname)
    data.to_excel(writer, "Sheet1", index=False)
    writer.save()
    if verbose:
        print("Saved data as: {}".format(fname))

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
        self.fname_needless_list = kwargs.get('fname_needless_list', os.path.join(cfg.root, cfg.fname_needless_list))

        self._manage_duplicate()
        self.stop_list = self._stop_list()
        self.synonym_pairs = self._synonym_pairs()
        self.needless_list = self._needless_list()

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

    def _synonym_pairs(self):
        with open(self.fname_synonyms, 'r', encoding='utf-8') as f:
            synonym_data = f.read()
        synonym_list = re.sub('\ufeff', '', synonym_data.strip())
        synonym_pairs = [tuple(pair.split('  ')) for pair in synonym_list.split('\n')]
        return synonym_pairs

    def synonym(self, text):
        if any([True for l, r in self.synonym_pairs if l in text]):
            for l, r in self.synonym_pairs:
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

    def _needless_list(self):
        with open(self.fname_needless_list, 'r', encoding='utf-8') as f:
            needless_list = f.read().replace('\n', '|')
        return needless_list

    def _mark_needless(self, text):
        
        return re.sub(needless_list, ' NEEDLESS ', text)

    def drop_needless(self, text):
        marked_sent = self.tokenize(re.sub(self.needless_list, ' NEEDLESS ', text))
        new_sent = []
        needless_index_list = [i for i, w in enumerate(marked_sent) if w == 'NEEDLESS']

        if needless_index_list:
            for idx in range(len(needless_index_list)):
                if idx == 0:
                    new_sent.extend(marked_sent[:needless_index_list[idx]])
                else:
                    i = needless_index_list[idx-1]
                    j = needless_index_list[idx]
                    if (j-i) < 3:
                        pass # NEEDLESS의 간격이 3단어 미만이면 -> 이건 광고!
                    else:
                        new_sent.extend(marked_sent[i:j])

                if idx == (len(needless_index_list)-1):
                    if (len(marked_sent) - needless_index_list[idx]) > 10: # 마지막 NEEDLESS 이후 7단어보다 많이 남았으면
                        new_sent.extend(marked_sent[needless_index_list[idx]:])
                    else:
                        continue
        else:
            new_sent = marked_sent

        new_text = ' '.join([w for w in new_sent if not w == 'NEEDLESS'])
        return re.sub('[^0-9a-zA-Zㄱ-힣]', ' ', new_text)

KistecPreprocess()._manage_duplicate()