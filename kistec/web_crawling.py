#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import sys
import os
import re
import time
import math
import numpy as np
import pickle as pk
import pandas as pd
import random
from config import Config
import GetOldTweets3 as got
from tqdm import tqdm, tqdm_notebook
from datetime import datetime, timedelta
import requests
import itertools
import urllib.request
from urllib.parse import quote
from bs4 import BeautifulSoup
from collections import defaultdict

from kistec.object import *
from kistec.function import *

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

class KistecTweet:
    def tweet_crawling(self, query, start_date, end_date, time_lag=3, verbose=False):

        '''
        Usage:
        - query: 한강대교
        - start_date: '20181001'
        - end_date: '20191103'
        '''

        tweets = []

        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
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

class KistecNaverNews:
    def __init__(self, **kwargs):
        self.time_lag = np.random.normal(loc=kwargs.get('time_lag', 3.0), scale=1.0)
        self.headers = {'User-Agent': '''
            [Windows64,Win64][Chrome,58.0.3029.110][KOS] 
            Mozilla/5.0 Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
            Chrome/58.0.3029.110 Safari/537.36
            '''}

        self.query = kwargs.get('query', '')
        self.query_input = quote(self.query.encode('utf-8'))

        self.start_date = kwargs.get('start_date', '')
        self.end_date = kwargs.get('end_date', '')
        self.date_split = kwargs.get('date_split', 'year') # year, month, day
        self.samples = kwargs.get('samples', '')

        self.url_base = 'https://search.naver.com/search.naver?&where=news&query={}&sm=tab_pge&sort=1&photo=0&field=0&reporter_article=&pd=3&ds={}&de={}&docid=&nso=so:dd,p:from{}to{},a:all&mynews=0&start={}&refresh_start=0'
        self.article_url_list = kwargs.get('article_url_list', [])
        self.articles = ''

        self.fdir_articles = kwargs.get('fdir_articles', os.path.join(cfg.root, cfg.fdir_articles))
        self.fname_articles_excel = kwargs.get('fname_articles_excel', 'articles_{}_{}_{}.xlsx'.format(self.query, self.start_date, self.end_date))

        self._error_articles = []

        self.data = None

    def _split_date(self):
        start_year, start_month, start_day = self._date_format(self.start_date).split('.')
        end_year, end_month, end_day = self._date_format(self.end_date).split('.')
        date_list = []

        if self.date_split == 'year':
            duration = int(end_year) - int(start_year) + 1
            for i in range(duration):
                if i == 0:
                    start_date = self.start_date
                    end_date = ''.join((start_year, '12', '31'))
                    date_list.append((start_date, end_date))
                elif i == (duration-1):
                    start_date = ''.join((end_year, '01', '01'))
                    end_date = self.end_date
                    date_list.append((start_date, end_date))
                else:
                    start_date = ''.join((str(int(start_year)+i), '01', '01'))
                    end_date = ''.join((str(int(start_year)+i), '12', '31'))
                    date_list.append((start_date, end_date))
        return date_list

    def _date_format(self, date):
        return datetime.strptime(date, '%Y%m%d').strftime('%Y.%m.%d')

    def _last_page(self, start_date, end_date):
        start_idx = 1
        start_date_datetime = self._date_format(start_date)
        end_date_datetime = self._date_format(end_date)

        url_list_page = self.url_base.format(self.query_input,
                                             start_date_datetime,
                                             end_date_datetime,
                                             start_date,
                                             end_date,
                                             start_idx)

        req = urllib.request.Request(url=url_list_page, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')

        time.sleep(self.time_lag)

        text = soup.select('div.title_desc.all_my')[0].text
        last_page = int(re.sub(',', '', text.split('/')[1])[:-1].strip())
        return last_page

    def _parse_list_page(self, url_list_page):
        req = urllib.request.Request(url=url_list_page, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')

        time.sleep(self.time_lag)

        href = soup.select('dl dd a')
        return [h.attrs['href'] for h in href if h.attrs['href'].startswith('https://news.naver.com/')]

    def get_article_url_list(self):
        if len(self.article_url_list) == 0:
            print('Naver News: Parse URL List ...')
            for start_date, end_date in self._split_date():
                print('  >>Parsing ({} to {}) ...'.format(start_date, end_date))
                last_page = self._last_page(start_date, end_date)
                max_start_idx = int(round(last_page, -1)) + 1
                index_list = list(range(1, max_start_idx, 10))
                # 네이버는 최대 4000개까지만 제공함
                with tqdm(total=len(index_list)) as pbar:
                    for start_idx in index_list:
                        start_date_datetime = self._date_format(start_date)
                        end_date_datetime = self._date_format(end_date)

                        url_list_page = self.url_base.format(self.query_input,
                                                             start_date_datetime,
                                                             end_date_datetime,
                                                             start_date,
                                                             end_date,
                                                             start_idx)

                        self.article_url_list.extend(self._parse_list_page(url_list_page))
                        pbar.update(1)
        return self.article_url_list

    def _get_comment(self, url_article):
        comments = []

        oid = url_article.split("oid=")[1].split("&")[0]
        aid = url_article.split("aid=")[1]
        page = 1    
        comment_header = {
            'User-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
            'referer':url_article}

        while True :
            url_comment_api = 'https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&templateId=default_society&pool=cbox5&_callback=jQuery1707138182064460843_1523512042464&lang=ko&country=&objectId=news'+oid+'%2C'+aid+'&categoryId=&pageSize=20&indexSize=10&groupId=&listType=OBJECT&pageType=more&page='+str(page)+'&refresh=false&sort=FAVORITE' 
            r = requests.get(url_comment_api, headers = comment_header)
            comment_content = BeautifulSoup(r.content,'html.parser')    
            total_comment = str(comment_content).split('comment":')[1].split(',')[0]

            match = re.findall('"contents":"([^\*]*)","userIdNo"', str(comment_content))
            comments.append(match)

            if int(total_comment) <= ((page)*20):
                break
            else : 
                page += 1

        return list(itertools.chain(*comments))

    def _parse_article_page(self, url_article):
        req = urllib.request.Request(url=url_article, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')

        time.sleep(self.time_lag)

        try:
            article_title = soup.select('h3[id=articleTitle]')[0].text
        except:
            self._error_articles.append(url_article)
            article_title = 'none'

        try:
            article_date = re.sub('\.', '', soup.select('span[class=t11]')[0].text.split(' ')[0])
        except:
            self._error_articles.append(url_article)
            article_date = 'none'

        try:
            article_category = soup.select('em[class=guide_categorization_item]')[0].text
        except:
            self._error_articles.append(url_article)
            article_category = 'none'

        try:
            article_content = soup.select('div[id=articleBodyContents]')[0].text.strip()
        except:
            self._error_articles.append(url_article)
            article_content = 'none'

#         try:
#             article_likeit_count_good = str(soup.select('li.u_likeit_list.good span.u_likeit_list_count._count')[0].text)
#         except:
#             self._error_articles.append(url_article)
#             article_likeit_count_good = 'none'

#         try:
#             article_likeit_count_warm = str(soup.select('li.u_likeit_list.warm span.u_likeit_list_count._count')[0].text)
#         except:
#             self._error_articles.append(url_article)
#             article_likeit_count_warm = 'none'

#         try:
#             article_likeit_count_sad = str(soup.select('li.u_likeit_list.sad span.u_likeit_list_count._count')[0].text)
#         except:
#             self._error_articles.append(url_article)
#             article_likeit_count_sad = 'none'

#         try:
#             article_likeit_count_angry = str(soup.select('li.u_likeit_list.angry span.u_likeit_list_count._count')[0].text)
#         except:
#             self._error_articles.append(url_article)
#             article_likeit_count_angry = 'none'

#         try:
#             article_likeit_count_want = str(soup.select('li.u_likeit_list.want span.u_likeit_list_count._count')[0].text)
#         except:
#             self._error_articles.append(url_article)
#             article_likeit_count_want = 'none'

        try:
            article_comment_list = self._get_comment(url_article)
            article_comment_count = len(article_comment_list)
        except:
            self._error_articles.append(url_article)
            article_comment_list = ['none']
            article_comment_count = 0
            
        article_config = {'url':url_article,
                          'title':article_title,
                          'date':article_date,
                          'category':article_category,
                          'content':article_content,

#                           'likeit_good':article_likeit_count_good,
#                           'likeit_warm':article_likeit_count_warm,
#                           'likeit_sad':article_likeit_count_sad,
#                           'likeit_angry':article_likeit_count_angry,
#                           'likeit_want':article_likeit_count_want,

                          'comment_list':article_comment_list,
                          'comment_count':article_comment_count
                          }

        return Article(**article_config)

    def get_articles(self):
        if len(self.articles) == 0:
            print('Naver News: Parse Articles ...')
            articles = []
            article_url_list = self.get_article_url_list()

            if self.samples:
                article_url_list = random.sample(article_url_list, self.samples)

            with tqdm(total=len(article_url_list)) as pbar:
                for idx, url in enumerate(article_url_list):
                    articles.append(self._parse_article_page(url))
                    pbar.update(1)

            self.articles = articles
        return self.articles

    def export_excel(self, articles):
        articles_dict = defaultdict(list)
        for article in articles:
            articles_dict['date'].append(article.date)
            articles_dict['title'].append(article.title)
            
            articles_dict['category'].append(article.category)
            articles_dict['content'].append(article.content)

            # articles_dict['likeit_good'].append(article.likeit_good)
            # articles_dict['likeit_warm'].append(article.likeit_warm)
            # articles_dict['likeit_sad'].append(article.likeit_sad)
            # articles_dict['likeit_angry'].append(article.likeit_angry)
            # articles_dict['likeit_want'].append(article.likeit_want)

            articles_dict['comment_list'].append(' SEP '.join(article.comment_list))
            articles_dict['comment_count'].append(article.comment_count)
            
            articles_dict['url'].append(article.url)

        articles_dict_sort = pd.DataFrame(articles_dict).sort_values(by=['date'], axis=0)
        fname_save = os.path.join(self.fdir_articles, self.fname_articles_excel)
        save_df2excel(articles_dict_sort, fname_save, verbose=True)

    def errors(self):
        if not self._error_articles:
            print('No error in articles')
            return None
        else:
            return list(set(self._error_articles))

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

def df2articles(fname_data):
    data = pd.DataFrame()
    with open(fname_data, 'rb') as f:
        data = pk.load(f)

    articles = []
    for idx in range(len(data)):
        doc = data.iloc[idx]
        article_config = {
            'date': doc['date'],
            'title': doc['title'],
            'category': doc['category'],
            'content': doc['content'],

            'comment_list': doc['comment_list'],
            'comment_count': doc['comment_count'],

            'url': doc['url']
            }
        articles.append(Article(**article_config))

    with open(fname_data, 'wb') as f:
        pk.dump(articles, f)