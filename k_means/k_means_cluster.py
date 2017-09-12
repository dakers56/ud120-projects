#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
import sys

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
minimum = None
maximum = None
print("printing data_dict")
from math import isnan

for k, v in data_dict.items():
    if isnan(float(v['salary'])):
        continue
    if minimum is None or maximum is None:
        minimum = v['salary']
        maximum = v['salary']
    else:
        if v['salary'] < minimum:
            minimum = v['salary']
        if v['salary'] > maximum:
            maximum = v['salary']

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)
i = 0

print("biggest value for exercised_stock_options: %s" % maximum)
print("smallest value for exercised_stock_options: %s" % minimum)

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()


### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

class KMeansClusterPoint:
    def __init__(self, point, cluster):
        self.point = point
        self.cluster = cluster


# Transforms data into KMeansClusterPoints
def get_initial_data_as_objects(feature_list):
    return [KMeansClusterPoint((f1, f2), 1) for f1, f2 in feature_list]


def euclid_dist(a, b):
    # print("Calculating euclidean distance")
    # print("type(a): %s" % type(a))
    # print("a: %s" % a)
    # print("type(b): %s" % type(b))
    # print("b: %s" % b)
    return (((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2)) ** .5


def distance_to_cluster(cluster, point):
    print("Calculating distance to cluster")
    print("cluster: %s" % cluster)
    print("point: %s" % point)
    return euclid_dist(cluster, point)


def min(a, b):
    return a if a < b else b


def closest_cluster(point, cluster_1, cluster_2):
    print("finding closest cluster")
    # print("point: %s" % type(point))
    # print("cluster_1: %s" % type(cluster_1))
    # print("cluster_2: %s" % type(cluster_2))
    # print("min: %s" % min(euclid_dist(point, cluster_1), euclid_dist(point, cluster_2)))
    d_to_cluster_1 = euclid_dist(point, cluster_1)
    d_to_cluster_2 = euclid_dist(point, cluster_2)
    lesser_d = min(d_to_cluster_1, d_to_cluster_2)
    if lesser_d == d_to_cluster_1:
        print("closer cluster is 1")
        return cluster_1
    print("closer cluster is 2")
    return cluster_2


def assign_to_cluster(data_point, cluster_1, cluster_2):
    closest = closest_cluster(data_point.point, cluster_1, cluster_2)
    # print("data_point.cluster: %s" % data_point.cluster)
    if closest is cluster_1:
        data_point.cluster = 1
    else:
        data_point.cluster = 2
        # print("data_point.cluster: %s" % data_point.cluster)


def get_random_cluster():
    import random
    return random.randint(0, 3), random.randint(0, 1000000)


def get_initial_clusters():
    return get_random_cluster(), get_random_cluster()


def shift_clusters(assigned_points):
    print("Shifting centroids")
    return find_new_centroid(assigned_points, 1), find_new_centroid(assigned_points, 2)


def find_new_centroid(pred, cluster_label):
    cluster_members = members_of_cluster(pred, cluster_label)
    return centroid(cluster_members)


def members_of_cluster(pred, cluster_label):
    cluster_members = []
    for p in pred:
        if p.cluster == cluster_label:
            cluster_members.append(p)
    return cluster_members


def centroid(list_of_points):
    total_x = 0.0
    total_y = 0.0
    for point in list_of_points:
        total_x += point.point[0]
        total_y += point.point[1]
    num_points = len(list_of_points)
    return total_x / num_points, total_y / num_points


def run_k_means(num_iters=100):
    print("Running k_means with num_iters = %s" % num_iters)
    k_means_feature_points = get_initial_data_as_objects(finance_features)
    cluster_1, cluster_2 = get_initial_clusters()
    for i in range(num_iters):
        print("Running iteration %s" % i)
        for point in k_means_feature_points:
            # print("Assigning %s to cluster" % (point.point))
            # print("type(point): %s" % type(point))
            # print("type(cluster_1): %s" % type(cluster_1))
            # print("type(cluster_2): %s" % type(cluster_2))
            assign_to_cluster(point, cluster_1, cluster_2)
            print("New cluster: %s" % point.cluster)
        cluster_1, cluster_2 = shift_clusters(k_means_feature_points)
        print("New centroids: %s, %s" % (cluster_1, cluster_2))
    return k_means_feature_points


pred = [k.cluster for k in run_k_means()]

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
