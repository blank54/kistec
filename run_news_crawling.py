#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("/data/blank54/workspace/kistec/src/")
from web_crawling import MyNaverNews

import os
import json
import dill
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

# User Input
queries = ["건물", "사고", "붕괴"]
start_dates = sorted(["20150101", "20150701", "20160101", "20160701", "20170101", 
    "20170701", "20180101", "20180701", "20190101", "20190701"], reverse=True)
end_dates = sorted(["20150630", "20151231", "20160630", "20161231", "20170630",
    "20171231", "20180630", "20181231", "20190630", "20191031"], reverse=True)

# Build Crawler
def build_crawler(queries, start_date, end_date, operate=False):
    query = "+".join(queries)

    fdir = os.getcwd()
    fname_crawler = "model/naver_news_crawler_{}_{}_{}.dill".format(query,
                                                                    start_date,
                                                                    end_date)

    if operate:
        naver_news = MyNaverNews(queries, start_date, end_date)
        with open(os.path.join(fdir, fname_crawler), "wb") as f:
            dill.dump(naver_news, f)

    else:
        with open(os.path.join(fdir, fname_crawler), "rb") as f:
            naver_news = dill.load(f)
    return naver_news

# Get Article URLs
def get_article_urls(naver_news, operate=False):
    fname_urls = "data/naver_news_urls_{}_{}_{}.json".format(naver_news.query,
                                                             naver_news.start_date,
                                                             naver_news.end_date)

    if operate:
        url_articles = naver_news.url_articles()

        with open(os.path.join(os.getcwd(), fname_urls), "w", encoding="utf-8") as f:
            json.dump(url_articles, f)
        print("\nSaved: Naver News URLs \n({})".format(fname_urls))
    else:
        with open(os.path.join(os.getcwd(), fname_urls), "r", encoding="utf-8") as f:
            url_articles = json.load(f)
        print("\nLoaded: Naver News URLs \n({})".format(fname_urls))
    return url_articles

# Get Articles
def get_article(naver_news, operate=False):
    fdir = os.getcwd()
    fname_article = "data/naver_news_articles_{}_{}_{}.xlsx".format(naver_news.query,
                                                                    naver_news.start_date,
                                                                    naver_news.end_date)

    if operate:
        df_naver = naver_news.articles()
        writer = pd.ExcelWriter(os.path.join(fdir, fname_article))
        df_naver.to_excel(writer, "Sheet1", index=False)
        writer.save()
        print("\nNaver News Crawling Complete: \n{} / {} to {}".format(naver_news.query,
                                                                   naver_news.start_date,
                                                                   naver_news.end_date))

    else:
        df_naver = pd.read_excel(os.path.join(fdir, fname_article))
        print("\nLoaded Naver News: \n{} / {} to {}".format(naver_news.query,
                                                                   naver_news.start_date,
                                                                   naver_news.end_date))

    return df_naver

# Web Crawling
for start_date, end_date in zip(start_dates, end_dates):
    naver_news = build_crawler(queries, start_date, end_date, operate=True)
    url_articles = get_article_urls(naver_news, operate=True)
    df_naver = get_article(naver_news, operate=True)