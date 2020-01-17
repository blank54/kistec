# Configuration
import sys
sys.path.append("/data/blank54/workspace/kistec/src/")
from embedding import TFIDF

from sklearn.decomposition import PCA
import pickle as pk

fname_tweet = "tweet_20191024"
fname_tfidf = "{}_tfidf.pk".format(fname_tweet)
with open(fname_tfidf, "rb") as f:
    tfidf_model = pk.load(f)

tfidf_vector = tfidf_model.tfidf_matrix
print(tfidf_vector.shape)

pca_model = PCA(n_components=100)
pca_vector = pca_model.fit_transform(tfidf_vector)
print(pca_vector.shape)

fname_tfidf_pos = "{}_tfidf_pos.pk".format(fname_tweet)
with open(fname_tfidf_pos, "wb") as f:
	pk.dump(pca_vector, f)