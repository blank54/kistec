#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re

class KistecPreprocess:
    def __init__(self, **kwargs):
        self.do_synonym = kwargs.get('do_synonym', True)
        self.fname_userword = kwargs.get('fname_userword', './kistec/thesaurus/userword.txt')
        self.fname_userdic = kwargs.get('fname_userdic', './kistec/thesaurus/userdic.txt')
        self.fname_stop = kwargs.get('fname_stop', './kistec/thesaurus/stop_list.txt')
        self.fname_synonyms = kwargs.get('fname_synonyms', './kistec/thesaurus/synonyms.txt')
        self.stop_list = self._stop_list()

    def _stop_list(self):
        with open(self.fname_stop, 'r', encoding='utf-8') as f:
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