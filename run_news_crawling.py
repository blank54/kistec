#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
from kistec.web_crawling import *
import os
import itertools
import pickle as pk
from config import Config

with open('./kistec/custom.cfg', 'r') as f:
    cfg = Config(f)

# Query
def get_query(fname_query):
    with open(fname_query, 'r', encoding='utf-8') as f:
        query_data = f.read()

    query_groups = [[query for query in group.split('\n')] for group in query_data.split('\n\n')]
    query_list = list(itertools.product(*query_groups))
    return ['+'.join(queries) for queries in query_list]

# Crawling
def do_crawling(query, start_date, end_date):
    fname_url_list = os.path.join(cfg.root, cfg.fdir_url_list_news, '{}_{}_{}.pk'.format(query, start_date, end_date))
    makedir(fname_url_list)

    crawling_config = {
        'query': query,
        'start_date': start_date,
        'end_date': end_date
        }
    news_crawler = NewsCrawler(**crawling_config)

    print('Naver News: Parse URL List ({}) ...'.format(query))
    url_list = news_crawler.get_url_list()
    with open(fname_url_list, 'wb') as f:
        pk.dump(url_list, f)
    # with open(fname_url_list, 'rb') as f:
    #     url_list = pk.load(f)
    print(len(url_list))

    print('Naver News: Parse Articles ({}) ...'.format(query))
    news_crawler.url_list = url_list
    articles = news_crawler.get_articles()
    print(len(articles))

    print('Web Crawling Done: {}_{}_{}'.format(query, start_date, end_date))

# main
def main():
    fname_query = os.path.join(cfg.root, cfg.fdir_query_list, 'query_list_20200123.txt')
    query_list = get_query(fname_query)
    start_date = '20100101'
    end_date = '20191231'

    for query in query_list:
        do_crawling(query, start_date, end_date)

main()