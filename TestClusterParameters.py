from builtins import staticmethod

import requests
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import ClusterByUrl
import json


class TestClusterParameters:

    @staticmethod
    def get_results_dict(urls):
        results = {}
        for url in urls:
            results[url] = {}
        return results

    @staticmethod
    def results_dict_to_array(dict):
        array = np.array([[dict[url][method] for method in sorted(dict[url])] for url in dict])
        return array

    @staticmethod
    def get_means(results):
        results_matrix = TestClusterParameters.results_dict_to_array(results)
        means = np.mean(results_matrix, axis=0)
        return means

    @staticmethod
    def select_method(methods, results):
        means = TestClusterParameters.get_means(results)
        method_index = np.argmax(means, axis=0)
        return methods[method_index] + ": " + str(means[method_index])


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    # get sample urls with ground truth values from json file
    with open("samples.json", 'r', encoding='utf-8') as urlfile:
        json_urls = json.load(urlfile)
    urls = list(json_urls.keys())
    print(str(len(urls)) + " urls")

    # test all methods available
    methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    # test different weights
    weights = [(0,1),(0.05,0.95),(0.1,0.9),(0.15,0.85),(0.2,0.8),(0.25,0.75),(0.3,0.7),(0.35,0.65),(0.4,0.6)]

    # search optimal parameters for each weight combination
    for w_tw, w_tfidf in weights:
        # get dictionaries to save 3 different evaluation scores:
        # v-measure, adjusted rand index and adjusted mutual info
        v_measure_results = TestClusterParameters.get_results_dict(urls)
        adj_rand_results = TestClusterParameters.get_results_dict(urls)
        adj_mutual_info_results = TestClusterParameters.get_results_dict(urls)
        fowlkes_mallows_results = TestClusterParameters.get_results_dict(urls)

        # for each url
        for url in urls:
            # query tweets by url using the ContexTweet Back-End API
            params = {'url': url}
            tweets_resp = requests.get("http://localhost:65500/tweets/byurl", params=params)
            tweets = tweets_resp.json()
            n_tweets = len(tweets)

            # get ordered texts
            texts = [tweet["text"] for tweet in tweets]
            # get TF-IDF features
            tfidf_feats = ClusterByUrl.ClusterByUrl.get_tfidf_features(texts)
            # get tweet features
            tweet_feats = ClusterByUrl.ClusterByUrl.get_tweet_features(tweets)

            # compute and combine distances
            dists = ClusterByUrl.ClusterByUrl.custom_distances(tweet_feats, tfidf_feats, w_tw, w_tfidf)

            # execute hierarchical clustering for each method
            for method in methods:
                # get linkage matrix from hierarchical clustering algorithm
                Z = linkage(dists, method=method)

                # uncomment this line to plot dendrograms
                # careful! will plot one dendrogram for each method and url!
                #ClusterByUrl.plot_dendrogram(Z, 2)

                # find k by elbow method
                k = ClusterByUrl.ClusterByUrl.calc_n_clusters(Z)

                # cluster observations according to estimated k
                clusters = fcluster(Z, k, 'maxclust')

                # calculate v-measure, rand index and adjusted mutual info scores
                v_measure = v_measure_score(json_urls[url], clusters)
                adj_rand = adjusted_rand_score(json_urls[url], clusters)
                adj_mutual_info = adjusted_mutual_info_score(json_urls[url], clusters)
                fowlkes_mallows = fowlkes_mallows_score(json_urls[url], clusters)

                # add results to the dictionaries, add parameter k in the method name for reference
                v_measure_results[url][method + "- k=" + str(k)] = v_measure
                adj_rand_results[url][method + "- k=" + str(k)] = adj_rand
                adj_mutual_info_results[url][method + "- k=" + str(k)] = adj_mutual_info
                fowlkes_mallows_results[url][method + "- k=" + str(k)] = fowlkes_mallows

        # after processing all the urls, determine the best method for each evaluation score!
        method_vmeasures = TestClusterParameters.select_method(methods, v_measure_results)
        method_adjrands = TestClusterParameters.select_method(methods, adj_rand_results)
        method_adjmutinf = TestClusterParameters.select_method(methods, adj_mutual_info_results)
        method_fowlkesmallows = TestClusterParameters.select_method(methods, fowlkes_mallows_results)

        print("Results for weights " + str(w_tw) + " - " + str(w_tfidf))
        print("v measures")
        print(method_vmeasures)
        print("adjusted rand index")
        print(method_adjrands)
        print("adjusted mutual information")
        print(method_adjmutinf)
        print("fowlkes mallows index")
        print(method_fowlkesmallows)
        print("-------------------------------------------")
