from sklearn.cluster import KMeans
import pandas as pd


def run_clustering(mnist_tr, files, nclust):
    km = KMeans(n_clusters=nclust).fit_predict(mnist_tr)
    df = pd.DataFrame.from_dict({x: km[i] for i, x in enumerate(files)}.items())
    df.to_csv("output_cluster.csv")
    return df
