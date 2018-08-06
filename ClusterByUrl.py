from builtins import staticmethod

import requests
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.sparse import csr_matrix, hstack
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import csv


class ClusterByUrl:

    @staticmethod
    def get_tfidf_features(corpus):
        tfidf_vec = TfidfVectorizer(ngram_range=(1, 3))
        return tfidf_vec.fit_transform(corpus)

    @staticmethod
    def get_tweet_features(tweets):
        matrix = []
        for tweet in tweets:
            row = [float(tweet["userId"]),
                   float(tweet["sentimentScore"])]
            matrix.append(row)
        scaler = StandardScaler()
        scaled_mat = scaler.fit_transform(matrix)
        return ClusterByUrl.avoid_zero_vectors(csr_matrix(scaled_mat))

    @staticmethod
    def get_tfidf_distances(tfidf_feats):
        dist = pdist(tfidf_feats.todense(), "cosine")
        return dist

    @staticmethod
    def get_tweet_distances(tweet_feats):
        minmax_scaler = MinMaxScaler()  # normalise to be able to weight the distances
        dist = pdist(tweet_feats.todense(), "sqeuclidean")
        dist = minmax_scaler.fit_transform(dist.reshape(-1, 1))  # reshape to 2d to scale with minmax
        dist = dist.reshape(dist.shape[0], )  # reshape again to 1d to return
        return dist

    @staticmethod
    def avoid_zero_vectors(matrix):
        for row in matrix:
            if not np.any(row.todense()):
                row[-1] = float(0.0001)
        return matrix

    @staticmethod
    def custom_distances(tw_feats, tfidf_feats, weight_tw=0.1, weight_tfidf=0.9):
        dist_tw = ClusterByUrl.get_tweet_distances(tw_feats)
        dist_tfidf = ClusterByUrl.get_tfidf_distances(tfidf_feats)
        dists = (dist_tw * weight_tw) + (dist_tfidf * weight_tfidf)
        return dists

    @staticmethod
    def plot_dendrogram(Z, i):
        # calculate full dendrogram1
        f = plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram ' + str(i))
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        f.show()

    @staticmethod
    def calc_n_clusters(z, i=0, plot_graph=False):
        '''
        From https://goo.gl/SaqES2
        :param z: The linkage matrix.
        :param i: The index to include in the graph title.
        :param plot_graph: True to plot observations and acceleration in a graph, False otherwise.
        :return: the number of clusters according to the elbow method.
        '''
        # calculate based on the distance from the last 10 mergings
        last = z[-10:, 2]

        # 2nd derivative of the distance
        acceleration = np.diff(last, 2)
        acceleration_rev = acceleration[::-1]

        if plot_graph:  # show observations and acceleration by number of clusters
            last_rev = last[::-1]
            idxs = np.arange(1, len(last) + 1)
            g = plt.figure(figsize=(25, 10))
            plt.title('Elbow method ' + str(i))
            plt.xlabel('clusters (k)')
            plt.ylabel('distance')
            plt.plot(idxs, last_rev)
            plt.plot(idxs[:-2], acceleration_rev)
            g.show()
        k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
        return k

    @staticmethod
    def visualise_clusters(x, i, clusters):
        c = plt.figure(figsize=(10, 8))
        plt.title("Clusters " + str(i))
        plt.xlabel('User')
        plt.ylabel('Sentiment score')
        plt.scatter(x[:, 0], x[:, 1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
        c.show()

    @staticmethod
    def compute_clustoids(sq_distance, ids, clusters):
        '''
        From https://stackoverflow.com/a/39870085, computes clustoids from the given parameters:
        "A clustoid is an element of a cluster whose sum of distances to the other elements in the same cluster is the
        lowest."
        :param sq_distance: squared matrix of the distances
        :param ids: ids of the observations, ordered as in the squared distances matrix.
        :param clusters: list of cluster assignments for all the observations.
        :return: a list with the ids of the clustoids
        '''
        clustoids = []
        id_list = list(zip(ids, clusters))
        for c in range(clusters.min(), clusters.max() + 1):
            # create mask from the list of assignments for extracting submatrix of the cluster
            mask = np.array([1 if i == c else 0 for i in clusters], dtype=bool)
            # take the index of the column with the smallest sum of distances from the sum submatrix
            i = np.argmin(sum(sq_distance[:, mask][mask, :]))
            # extract ids of clusters from list of (id, cluster)
            id_sublist = [id for (id, cluster) in id_list if cluster == c]
            # get element closest to centroid indexing the sublist
            clustoids.append(id_sublist[i])
        return clustoids

    @staticmethod
    def append_to_csv_file(ids, url):
        data = []
        for id in ids:
            data.append([url, id])
        with open('data/ids-urls.txt', newline='', mode='a+', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
            writer.writerows(data)


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

    # query all distinct urls
    urls_resp = requests.get("http://localhost:65500/urls")
    urls = urls_resp.json()
    print(str(len(urls)) + " urls")

    # for each url
    for url in urls:
        # query tweets by url
        params = {'url': url}
        tweets_resp = requests.get("http://localhost:65500/tweets/byurl", params=params)
        tweets = tweets_resp.json()
        n_tweets = len(tweets)
        print(str(n_tweets) + " tweets for url " + url)

        # if the url has more than 10 tweets, apply clustering
        if n_tweets > 7:
            # get ordered ids and texts
            ids = [tweet["id"] for tweet in tweets]
            texts = [tweet["text"] for tweet in tweets]
            # get TF-IDF features
            tfidf_feats = ClusterByUrl.get_tfidf_features(texts)
            # get tweet features
            tweet_feats = ClusterByUrl.get_tweet_features(tweets)
            # combine features
            feats = hstack([tweet_feats, tfidf_feats])

            # compute and combine distances
            dists = ClusterByUrl.custom_distances(tweet_feats, tfidf_feats, 0.15, 0.85)

            # perform hierarchical clustering
            Z = linkage(dists, method='complete')

            # uncomment this line to plot dendrograms
            # careful! it plots len(url) dendrograms!
            #ClusterByUrl.plot_dendrogram(Z, 2)

            # find k by elbow method
            k = ClusterByUrl.calc_n_clusters(Z, 2)
            print("number of clusters: ", k)

            # cluster observations according to k
            clusters = fcluster(Z, k, 'maxclust')

            # uncomment this line to plot clusters
            # careful! it plots len(url) graphs
            #ClusterByUrl.visualise_clusters(feats.A, 2, clusters)

            # calculate clustoids
            clustoids = ClusterByUrl.compute_clustoids(squareform(dists), ids, clusters)

            # append to csv file for later import
            ClusterByUrl.append_to_csv_file(clustoids, url)

    print("Finished")
