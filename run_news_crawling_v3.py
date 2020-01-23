#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
from kistec.web_crawling import KistecNaverNews
import os
import itertools
import pickle as pk
from config import Config

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

# Query
fname_query = os.path.join(cfg.root, cfg.fdir_query_list, 'query_list_20200123.txt')
with open(fname_query, 'r', encoding='utf-8') as f:
    query_data = f.read()

query_groups = [[query for query in group.split('\n')] for group in query_data.split('\n\n')]
query_list = list(itertools.product(*query_groups))

# Build Crawler
def build_crawler(query_input):
    query, start_date, end_date = query_input
    fdir_articles = os.path.join(cfg.root, cfg.fdir_articles)
    os.makedirs(fdir_articles, exist_ok=True)

    print('Web Crawling: {}_{}_{}'.format(query, start_date, end_date))

    crawler_config = {'query': query,
                      'start_date': start_date,
                      'end_date': end_date,
                      'fdir_articles': fdir_articles}

    news_crawler = KistecNaverNews(**crawler_config)
    return news_crawler

# Parse List Page
def parse_list_page(query_input):
    query, start_date, end_date = query_input
    fdir_url_list = os.path.join(cfg.root, cfg.fdir_url_list_news)
    fname_url_list = 'url_list_{}_{}_{}.pk'.format(query, start_date, end_date)
    os.makedirs(fdir_url_list, exist_ok=True)

    if fname_url_list in os.listdir(fdir_url_list):
        with open(os.path.join(fdir_url_list, fname_url_list), 'rb') as f:
            article_url_list = pk.load(f)
    else:
        article_url_list = news_crawler.get_article_url_list()
        with open(os.path.join(fdir_url_list, fname_url_list), 'wb') as f:
            pk.dump(article_url_list, f)

    return article_url_list

# Parse Article Page
def parse_article_page(query_input):
    query, start_date, end_date = query_input
    
    fname_articles = 'articles_{}_{}_{}.pk'.format(query, start_date, end_date)
    if fname_articles in os.listdir(news_crawler.fdir_articles):
        with open(os.path.join(news_crawler.fdir_articles, fname_articles), 'rb') as f:
            articles = pk.load(f)
    else:
        articles = news_crawler.get_articles()
        with open(os.path.join(news_crawler.fdir_articles, fname_articles), 'wb') as f:
            pk.dump(articles, f)

    return articles

# main
for query_set in query_list:
    query = '+'.join(query_set)
    start_date = '20100101'
    end_date = '20141231'
    query_input = [query, start_date, end_date]

    news_crawler = build_crawler(query_input)

    article_url_list = parse_list_page(query_input)
    news_crawler.article_url_list = article_url_list
    articles = parse_article_page(query_input)
    news_crawler.articles = articles
    news_crawler.export_excel(articles)

    print('Web Crawling Done: {}_{}_{}'.format(query, start_date, end_date))