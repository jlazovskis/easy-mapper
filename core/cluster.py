###
### Clustering functions for easy-mapper
###

# PyClustering packages
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

# kmeans:
def cl_kmeans(sample):
	initial_centers = kmeans_plusplus_initializer(sample, 2).initialize()
	kmeans_instance = kmeans(sample, initial_centers)
	kmeans_instance.process()
	return kmeans_instance.get_clusters()

# kmedians:
def cl_kmedians(sample):
	initial_medians = [] # unfinished, unclear how to get medians
	kmedians_instance = kmedians(sample, initial_medians)
	kmedians_instance.process()
	return kmedians_instance.get_clusters()

# xmeans: 
def cl_xmeans(sample):
	initial_centers = kmeans_plusplus_initializer(sample, 2).initialize()
	xmeans_instance = xmeans(sample, initial_centers, 20)
	xmeans_instance.process()
	return xmeans_instance.get_clusters()

# slc: single linkage clustering