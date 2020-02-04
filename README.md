# KISTEC X C!LAB
Text Analysis of Facility Maintenance System
- - -
## _Project Information_
- Performed by C!LAB (@Seoul Nat'l Univ.)
- Supported by KISTEC
- Duration: 2019. - 2020.

## _Contributors_
- Project Director: Seokho Chi, _Ph.D._, _Associate Prof._ (shchi@snu.ac.kr, https://cm.snu.ac.kr/)
- Main Developer: Seonghyeon Boris Moon (blank54@snu.ac.kr, https://github.com/blank54/)
- Sub Developer: Taeyeon Chang (jgwoon1838@snu.ac.kr, _a.k.a. Kowoon Chang_)

- - -
## NewsCrawler
```
from config import Config
with open(FNAME_YOUR_CONFIG, 'r') as f:
    cfg = Config(f)

from kistec import NewsCrawler

input_query = YOUR_QUERY    # '교량+사고+유지관리'
date_from = YOUR_DATE_FROM  # '20190701'
date_to = YOUR_DATE_TO      # '20190705'

news_crawler = NewsCrawler(**crawling_config)

url_list = news_crawler.get_url_list()
articles = news_crawler.get_articles()
```
The results(i.e., **_url_list.pk_**, **_article.pk_**, **_articles.xlsx_**) would be saved as each **_fname_** in config
