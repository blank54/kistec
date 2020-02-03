#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import sys
import time
import math
import random
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from collections import defaultdict
from datetime import datetime, timedelta

import requests
import itertools
import urllib.request
from urllib.parse import quote
from bs4 import BeautifulSoup
import GetOldTweets3 as got

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
matplotlib.rc('font', family='NanumBarunGothic')

import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.ldamodel import LdaModel

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from config import Config
with open('/data/blank54/workspace/kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

class Article:
    def __init__(self, **kwargs):
        self.kfc = KistecFunction()

        self.id = kwargs.get('id', '')
        self.url = kwargs.get('url', '')
        self.url_uniq = self.kfc.get_url_uniq(self.url)
        self.title = kwargs.get('title', '')
        self.date = kwargs.get('date', '')
        self.category = kwargs.get('category', '')
        self.content = kwargs.get('content', '')
        self.content_prep = kwargs.get('content_prep', '')

        self.likeit_good = kwargs.get('likeit_good', '')
        self.likeit_warm = kwargs.get('likeit_warm', '')
        self.likeit_sad = kwargs.get('likeit_sad', '')
        self.likeit_angry = kwargs.get('likeit_angry', '')
        self.likeit_want = kwargs.get('likeit_want', '')

        self.comment_list = kwargs.get('comment_list', 'none')
        self.comment_count = kwargs.get('comment_count', 0)

        self.topic_id = kwargs.get('topic_id', '')

    def __call__(self):
        return self.url

    def __str__(self):
        return '{}: {}'.format(self.id, self.url)

    def __len__(self):
        return len(self.content)

class KistecFunction:
    def makedir(self, path):
        if path.endswith('/'):
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

    def save_df2excel(self, data, fname, verbose=False):
        self.makedir(fname)

        writer = pd.ExcelWriter(fname)
        data.to_excel(writer, "Sheet1", index=False)
        writer.save()
        if verbose:
            print("Saved data as: {}".format(fname))

    def flist_archive(self, fdir):
        flist = []
        for (path, _, files) in os.walk(fdir):
            flist.extend([os.path.join(path, file) for file in files if file.endswith('.pk')])
        return flist

    def get_url_uniq(self, url):
        return url.split('/')[-1]

class KistecPreprocess:
    def __init__(self, **kwargs):
        self.do_synonym = kwargs.get('do_synonym', cfg.do_synonym)
        self.fname_userword = kwargs.get('fname_userword', os.path.join(cfg.root, cfg.fname_userword))
        self.fname_userdic = kwargs.get('fname_userdic', os.path.join(cfg.root, cfg.fname_userdic))
        self.fname_stop_list = kwargs.get('fname_stop_list', os.path.join(cfg.root, cfg.fname_stop_list))
        self.fname_synonyms = kwargs.get('fname_synonyms', os.path.join(cfg.root, cfg.fname_synonyms))
        self.fname_needless_list = kwargs.get('fname_needless_list', os.path.join(cfg.root, cfg.fname_needless_list))

        self.stop_list = self.__read_stop_list()
        self.synonym_pairs = self.__read_synonym_pairs()
        self.needless_list = self.__read_needless_list()

    def __call__(self):
        self.__manage_duplicate()
        self.__build_userdic()

    def __manage_duplicate(self):
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

    def __read_stop_list(self):
        with open(self.fname_stop_list, 'r', encoding='utf-8') as f:
            return sorted(set(f.read().strip().split('\n')), reverse=False)

    def __build_userdic(self):
        with open(self.fname_userword, 'r', encoding='utf-8') as f:
            wordlist = f.read().strip().split('\n')
            if not wordlist:
                return None
            userdic = '\n'.join([str(w+'\tNNP') for w in wordlist if len(w)>0])
        with open(self.fname_userdic, 'w', encoding='utf-8') as f:
            f.write(re.sub('\ufeff', '', userdic))

    def __read_synonym_pairs(self):
        with open(self.fname_synonyms, 'r', encoding='utf-8') as f:
            synonym_data = f.read()
        synonym_list = re.sub('\ufeff', '', synonym_data.strip())
        synonym_pairs = [tuple(pair.split('  ')) for pair in synonym_list.split('\n')]
        return synonym_pairs

    def __read_needless_list(self):
        with open(self.fname_needless_list, 'r', encoding='utf-8') as f:
            needless_list = f.read().replace('\n', '|')
        return needless_list

    def synonym(self, text):
        if any([True for l, r in self.synonym_pairs if l in text]):
            for l, r in self.synonym_pairs:
                text = re.sub(l, r, text)
        return text

    def tokenize(self, text):
        if self.do_synonym:
            text = self.synonym(text)

        cleaned_text = text
        tokenized_text = [w.strip() for w in re.split(' |  |\n', cleaned_text) if len(w)>0]
        return tokenized_text

    def stopword_removal(self, text, return_type='list'):
        result = [w for w in self.tokenize(text) if w not in self.stop_list]
        if return_type == 'list':
            return result
        elif return_type == 'str':
            return ' '.join(result)

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
                    if (j-i) < cfg.distance_of_needless:
                        pass # NEEDLESS의 간격이 일정 단어 미만이면 -> 이건 광고!
                    else:
                        new_sent.extend(marked_sent[i:j])

                if idx == (len(needless_index_list)-1):
                    if (len(marked_sent) - needless_index_list[idx]) > cfg.content_left: # 마지막 NEEDLESS 이후 일정 단어보다 많이 남았으면
                        new_sent.extend(marked_sent[needless_index_list[idx]:])
                    else:
                        continue
        else:
            new_sent = marked_sent

        new_text = ' '.join([w for w in new_sent if not w == 'NEEDLESS'])
        return re.sub('[^0-9a-zA-Zㄱ-힣]', ' ', new_text)

class Run:
    def __init__(self):
        self.kpr = KistecPreprocess()

    def run_embedding_word2vec(self, tokenized_docs, parameters, verbose=True):
        '''
        Input docs: [[word, word, ...], [word, word, ...], ...]
        '''

        model = Word2Vec(
            size=parameters.get('size', cfg.w2v_vector_size),
            window=parameters.get('window', cfg.w2v_window),
            min_count=parameters.get('min_count', cfg.w2v_min_count),
            workers=parameters.get('workers', cfg.w2v_workers),
            sg=parameters.get('architecture', cfg.w2v_architecture),
            alpha=parameters.get('alpha', cfg.w2v_alpha),
            min_alpha=parameters.get('min_alpha', cfg.w2v_min_alpha)
            )
        model.build_vocab(tokenized_docs)
        
        max_epoch = parameters.get('max_epoch', cfg.w2v_max_epoch)
        alpha_step = parameters.get('alpha_step', cfg.w2v_alpha_step)
        with tqdm(total=max_epoch) as pbar:
            for epoch in range(max_epoch):
                model.train(
                    sentences=tokenized_docs,
                    total_examples=model.corpus_count,
                    epochs=epoch
                    )
                model.alpha -= alpha_step
                if verbose:
                    pbar.update(1)
        return model

    def run_embedding_doc2vec(self, docs_with_tag, parameters, verbose=True):
        '''
        Input docs: [(tag, doc), (tag, doc), ...]
        '''

        docs_for_d2v = [TaggedDocument(words=doc, tags=[tag]) for tag, doc in docs_with_tag]
        parameters = kwargs.get('parameters', {})
            
        model = Doc2Vec(
            vector_size=parameters.get('vector_size', cfg.d2v_vector_size),
            alpha=parameters.get('alpha', cfg.d2v_alpha),
            min_alpha=parameters.get('min_alpha', cfg.d2v_min_alpha),
            min_count=parameters.get('min_count', cfg.d2v_min_count),
            window=parameters.get('window', cfg.d2v_window),
            workers=parameters.get('workers', cfg.d2v_workers),
            dm=parameters.get('dm', cfg.d2v_dm)
            )
        model.build_vocab(docs_for_d2v)

        max_epoch = parameters.get('max_epoch', cfg.d2v_max_epoch)
        alpha_step = parameters.get('alpha_step', cfg.d2v_alpha_step)
        with tqdm(total=max_epoch) as pbar:
            for epoch in range(max_epoch):
                model.train(
                    documents=docs_for_d2v,
                    total_examples=model.corpus_count,
                    epochs=epoch)
                model.alpha -= alpha_step
                if verbose:
                    pbar.update(1)
        return model

    def run_crawling_news(self, input_query, date_from, date_to):
        '''
        input_query: '교량+사고+유지관리'
        date_from: '20150101'
        date_to: '20191130'
        '''

        crawling_config = {
            'query': NewsQuery(input_query),
            'date_from': NewsDate(date_from),
            'date_to': NewsDate(date_to)
            }
        news_crawler = NewsCrawler(**crawling_config)

        url_list = news_crawler.get_url_list()
        print(len(url_list))

        articles = news_crawler.get_articles()
        print(len(articles))

class NewsCrawler:
    def __init__(self, **kwargs):
        self.kfc = KistecFunction()

        self.time_lag = np.random.normal(loc=kwargs.get('time_lag', 3.0), scale=1.0)
        self.headers = {'User-Agent': '''
            [Windows64,Win64][Chrome,58.0.3029.110][KOS] 
            Mozilla/5.0 Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
            Chrome/58.0.3029.110 Safari/537.36
            '''}

        self.query = NewsQuery(kwargs.get('input_query', ''))
        self.date_from = NewsDate(kwargs.get('date_from', ''))
        self.date_to = NewsDate(kwargs.get('date_to', ''))
        self.query_info = '{}_{}_{}'.format(self.query.query, self.date_from.date, self.date_to.date)

        self.do_sampling = kwargs.get('do_samples', cfg.news_crawling_do_sampling)
        self.news_num_samples = kwargs.get('num_samples', cfg.news_num_samples)

        self.url_base = 'https://search.naver.com/search.naver?&where=news&query={}&sm=tab_pge&sort=1&photo=0&field=0&reporter_article=&pd=3&ds={}&de={}&docid=&nso=so:dd,p:from{}to{},a:all&mynews=0&start={}&refresh_start=0'
        self.url_start = 'https://news.naver.com/'
        self.url_list = kwargs.get('url_list', [])
        self.articles = kwargs.get('articles', [])

        self.fname_news_url_list = kwargs.get('fname_news_url_list', os.path.join(cfg.root, cfg.fdir_news_url_list, '{}.pk'.format(self.query_info)))
        self.fdir_news_data_articles = kwargs.get('fdir_news_data_articles', os.path.join(cfg.root, cfg.fdir_news_data_articles, '{}').format(self.query.query))
        self.fdir_news_corpus_articles = kwargs.get('fdir_news_corpus_articles', os.path.join(cfg.root, cfg.fdir_news_corpus_articles, '{}').format(self.query.query))
        self.news_archive = kwargs.get('news_archive', self.kfc.flist_archive(self.fdir_news_corpus_articles))
        self.fname_articles = kwargs.get('fname_articles', os.path.join(self.fdir_news_data_articles, '{}_{}_{}.xlsx'.format(self.query.query, self.date_from.date, self.date_to.date)))

        self._errors = []
        self.fname_errors = kwargs.get('fname_errors', os.path.join(cfg.root, cfg.fdir_news_errors))

    def __get_last_page(self):
        start_idx = 1
        url_list_page = self.url_base.format(self.query(),
                                             self.date_from.formatted,
                                             self.date_to.formatted,
                                             self.date_from.date,
                                             self.date_to.date,
                                             start_idx)

        req = urllib.request.Request(url=url_list_page, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(self.time_lag)

        _text = soup.select('div.title_desc.all_my')[0].text
        last_page = int(re.sub(',', '', _text.split('/')[1])[:-1].strip())
        return last_page

    def __parse_list_page(self, url_list_page):
        req = urllib.request.Request(url=url_list_page, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(self.time_lag)
        href = soup.select('dl dd a')
        return [h.attrs['href'] for h in href if h.attrs['href'].startswith(self.url_start)]

    def get_url_list(self):
        print('  >>Parsing ({} to {}) ...'.format(self.date_from.date, self.date_to.date))
        if os.path.isfile(self.fname_news_url_list):
            with open(self.fname_news_url_list, 'rb') as f:
                url_list = pk.load(f)
        else:
            url_list = []
            last_page = self.__get_last_page()
            max_start_idx = int(round(last_page, -1)) + 1
            index_list = list(range(1, max_start_idx, 10)) # 네이버는 최대 4000개까지만 제공함
            with tqdm(total=len(index_list)) as pbar:
                for start_idx in index_list:
                    url_list_page = self.url_base.format(self.query(),
                                                         self.date_from.formatted,
                                                         self.date_to.formatted,
                                                         self.date_from.date,
                                                         self.date_to.date,
                                                         start_idx)
                    url_list.extend(self.__parse_list_page(url_list_page))
                    pbar.update(1)

            self.kfc.makedir(fname_news_url_list)
            with open(self.fname_news_url_list, 'wb') as f:
                pk.dump(url_list, f)
        return url_list

    def __parse_comment(self, url_article):
        comments = []

        oid = url_article.split("oid=")[1].split("&")[0]
        aid = url_article.split("aid=")[1]
        page = 1    
        comment_header = {
            'User-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
            'referer':url_article}

        while True:
            url_comment_api = 'https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&templateId=default_society&pool=cbox5&_callback=jQuery1707138182064460843_1523512042464&lang=ko&country=&objectId=news'+oid+'%2C'+aid+'&categoryId=&pageSize=20&indexSize=10&groupId=&listType=OBJECT&pageType=more&page='+str(page)+'&refresh=false&sort=FAVORITE' 
            r = requests.get(url_comment_api, headers=comment_header)
            comment_content = BeautifulSoup(r.content,'html.parser')    
            total_comment = str(comment_content).split('comment":')[1].split(',')[0]
            match = re.findall('"contents":"([^\*]*)","userIdNo"', str(comment_content))
            comments.append(match)

            if int(total_comment) <= ((page)*20):
                break
            else : 
                page += 1

        return list(itertools.chain(*comments))

    def __parse_article_page(self, url_article):
        req = urllib.request.Request(url=url_article, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(self.time_lag)

        try:
            article_title = soup.select('h3[id=articleTitle]')[0].text
        except:
            self._errors.append(url_article)
            article_title = 'NoTitle'

        try:
            article_date = re.sub('\.', '', soup.select('span[class=t11]')[0].text.split(' ')[0])
        except:
            self._errors.append(url_article)
            article_date = 'NoDate'

        try:
            article_category = soup.select('em[class=guide_categorization_item]')[0].text
        except:
            self._errors.append(url_article)
            article_category = 'NoCategory'

        try:
            article_content = soup.select('div[id=articleBodyContents]')[0].text.strip()
        except:
            self._errors.append(url_article)
            article_content = 'NoContent'

        try:
            article_comment_list = self.__parse_comment(url_article)
            article_comment_count = len(article_comment_list)
        except:
            self._errors.append(url_article)
            article_comment_list = ['NoComments']
            article_comment_count = 0

        article_config = {
            'url': url_article,
            'title': article_title,
            'date': article_date,
            'category': article_category,
            'content': article_content,
            'comment_list': article_comment_list,
            'comment_count': article_comment_count
            }
        return Article(**article_config)

    def get_articles(self):
        with open(self.fname_news_url_list, 'rb') as f:
            url_list = pk.load(f)

        if self.do_sampling:
            url_list = random.sample(url_list, self.news_num_samples)
        else:
            pass

        articles = []
        print('')
        with tqdm(total=len(url_list)) as pbar:
            for idx, url in enumerate(url_list):
                if any((self.kfc.get_url_uniq(url) in url_) for url_ in self.news_archive):
                    fname_article_in_archive = [f for f in self.news_archive if self.kfc.get_url_uniq(url) in f][0]
                    with open(fname_article_in_archive, 'rb') as f:
                        article = pk.load(f)
                        article.id = ''
                else:
                    article = self.__parse_article_page(url)

                fname_article = os.path.join(self.fdir_news_corpus_articles, article.date, '{}.pk'.format(article.url_uniq))
                self.kfc.makedir(fname_article)
                with open(fname_article, 'wb') as f:
                    pk.dump(article, f)
                articles.append(article)
                pbar.update(1)
        self.__export_excel(articles)

        if self._errors:
            print('Errors: {}'.format(len(self._errors)))
            with open(self.fname_errors, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self._errors))

        return articles

    def __export_excel(self, articles):
        articles_dict = defaultdict(list)
        for article in articles:
            articles_dict['id'].append(article.id)
            articles_dict['date'].append(article.date)
            articles_dict['title'].append(article.title)
            articles_dict['category'].append(article.category)
            articles_dict['content'].append(article.content)
            articles_dict['comment_list'].append(' SEP '.join(article.comment_list))
            articles_dict['comment_count'].append(article.comment_count)
            articles_dict['url'].append(article.url)

        articles_dict_sort = pd.DataFrame(articles_dict).sort_values(by=['date'], axis=0)
        self.kfc.save_df2excel(articles_dict_sort, self.fname_articles)

class NewsQuery:
    def __init__(self, query):
        self.query = query

    def __call__(self):
        return quote(self.query.encode('utf-8'))

    def __str__(self):
        return '{}'.format(self.query)

    def __len__(self):
        return len(self.query.split('+'))

class NewsDate:
    def __init__(self, date):
        self.date = date
        self.formatted = self.__convert_date()

    def __call__(self):
        return self.formatted

    def __str__(self):
        return '{}'.format(self.__call__())

    def __convert_date(self):
        try:
            return datetime.strptime(self.date, '%Y%m%d').strftime('%Y.%m.%d')
        except:
            return ''

class KistecTweet:
    def date_from(self, query, date_from, date_to, time_lag=3, verbose=False):

        '''
        Usage:
        - query: 한강대교
        - date_from: '20181001'
        - date_to: '20191103'
        '''

        tweets = []

        start = datetime.strptime(date_from, '%Y%m%d')
        end = datetime.strptime(date_to, '%Y%m%d')
        days = [start+timedelta(days=d) for d in range(0, (end-start).days)]
        for day in days:
            d = day.strftime('%Y-%m-%d')
            d_until = datetime.strftime(day + timedelta(days=1), '%Y-%m-%d')

            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                                       .setSince(d)\
                                                       .setUntil(d_until)\
                                                       .setMaxTweets(-1)

            tweet_data = got.manager.TweetManager.getTweets(tweetCriteria)
            print('Query: {} / Date: {}'.format(query, d))

            if verbose:
                running_time = time_lag*len(tweet_data)
                now = datetime.now()
                then = now + timedelta(seconds=running_time)
                print('Expected Running Time: {} sec ({} min)'.format(running_time, np.round(running_time/60, 2)))
                print('Start: {} | End: {}'.format(now.strftime('%Y-%m-%d %H:%M:%S'), then.strftime('%Y-%m-%d %H:%M:%S')))

            for idx, tweet in enumerate(tqdm_notebook(tweet_data)):
                username = tweet.username
                tweet_date = tweet.date.strftime('%Y-%m-%d')
                tweet_time = tweet.date.strftime('%H:%M:%S')
                content = tweet.text
                link = tweet.permalink
                retweets = tweet.retweets
                favorites = tweet.favorites
                
                tweet_key = '_'.join((username, tweet_date, tweet_time))
                result_list = [tweet_key, username, tweet_date, tweet_time, content, link, retweets, favorites]
                tweets.append(result_list)

                sys.stdout.write('\rParsing Tweets: {}/{}'.format((idx+1), len(tweet_data)))
                time.sleep(time_lag)

        return pd.DataFrame(tweets, columns=['key', 'username', 'date', 'time', 'content', 'link', 'retweets', 'favorites'])

    def tweet_concat(self, fdir, fname=None, drop_link=True):
        flist = [f for f in os.listdir(fdir) if f.endswith('.xlsx')]
        sheet = [pd.read_excel(os.path.join(fdir, file)) for file in flist]
        data = pd.concat(sheet, axis=0, join='outer', sort=True)

        if drop_link:
            data = data.drop(columns=['link'])

        if not fname:
            today = datetime.strftime(datetime.now(), '%Y%m%d')
            fname = './tweet_{}.xlsx'.format(today)

        writer = pd.ExcelWriter(os.path.join(fdir, fname))
        data.to_excel(writer, 'Sheet1', index=False)
        writer.save()
        print('Tweet Concatenation Complete: {}'.format(os.path.join(fdir, fname)))
        return data

def load_crawled_data(fdir, match, pattern):
    if match == 'start':
        flist = [os.path.join(fdir, fname) for fname in os.listdir(fdir)
                    if fname.startswith(pattern)]
    elif match == 'end':
        flist = [os.path.join(fdir, fname) for fname in os.listdir(fdir)
                    if fname.endswith(pattern)]

    sheet = [pd.read_excel(file) for file in flist]
    data = pd.concat(sheet, axis=0, join='outer')
    return data

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
