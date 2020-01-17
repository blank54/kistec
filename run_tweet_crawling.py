# Configuration
import sys
sys.path.append("/data/blank54/workspace/kistec/src/")
from web_crawling import MyTweet

import os
import json
import pandas as pd

mytweet = MyTweet()

# User Input
# query = sys.argv[1]
# queries = ["교량", "터널", "건물", "건축물", "상하수도", "상수도", "하수도",
#     "댐", "저수지", "제방"]
queries = ["건물"]

start_date = "20180101"
end_date = "20180105"

for query in queries:
	# Tweet Crawling
	tweet_result = mytweet.tweet_crawling(query, start_date, end_date)

	# Save Results
	fdir = os.getcwd()
	fname = "data/tweet_crawling_{}_{}_{}.xlsx".format(query, start_date, end_date)
	writer = pd.ExcelWriter(os.path.join(fdir, fname))
	tweet_result.to_excel(writer, "Sheet1", index=False)
	writer.save()

	print("Tweet Crawling Complete: {} / {} to {}".format(query, start_date, end_date))