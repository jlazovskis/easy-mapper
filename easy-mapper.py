#!/usr/bin/env python

# Start message 1
print('easy-mapper v1\n' + '-'*40 + '\nLoading packages...',end='')

import argparse                  # parsing input argument
import numpy as np               # storing and working with vectors
import pandas as pd              # storing data
from math import isclose         # fixing rounding errors
from math import sqrt            # drawing nodes at the right size
import itertools                 # working with lists
import warnings                  # supressing warnings
import networkx as nx            # creating graphs
import matplotlib.pyplot as plt  # drawing graphs
from matplotlib import cm        # coloring graphs

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

# Start message 2
print('done.\nReading and parsing input...',end='')

###
### Setup
###

# Description of script and arguments
parser = argparse.ArgumentParser(
    description='Easy implementation of mapper.',
    usage='python easy-mapper.py data [--function=f] [--intervals=i] [--overlap=o] [--ids=True/False')
parser.add_argument('datafile', type=str, help='The input file of data, each line containing coordinates separated by spaces')
parser.add_argument('--function', type=str, default='3nearest', help='Parameter 1: The real-valued filter function that takes in the data. Default is 3 nearest neighbors.')
parser.add_argument('--intervals', type=int, default=10, help='Parameter 2: The number of intervals for creating a cover of the range of the filter function. Default is 10.')
parser.add_argument('--overlap', type=float, default=0.1, help='Parameter 3: The percentage overlap for successive intervals. Default is 10%.')
parser.add_argument('--ids', type=bool, default=False, help='Whether or not there are IDs for each point in the input file. Default is False.')
parser.add_argument('--output', type=str, default='mpl', help='Type of output for the graph, can be mpl for matplotlib or txt for a text file. Default is mpl.')
args = parser.parse_args()

# Load input file and make dataframe of vectors
f = open(args.datafile, 'r')
if args.ids:
	data_temp = list(map(lambda x: x[:-2].split(' '), f.readlines()))
	data_raw = list(map(lambda x: [x[0],np.array(x[1:],dtype=float)], data_temp))
	data = pd.DataFrame(data_raw, columns=['id','vec'])
if not args.ids:
	data_raw = list(map(lambda x: np.array(list(map(lambda y: float(y), x[:-2].split(' '))),dtype=float), f.readlines()))
	data = pd.DataFrame([[str(i),data_raw[i]] for i in range(len(data_raw))], columns = ['id','vec'])
f.close() 

# Create distance matrix
n = len(data_raw)
M = np.array([np.array(np.zeros(n),dtype=float) for i in range(n)])
for v1_ind in range(n):
	for v2_ind in range(v1_ind+1,n):
		d = np.linalg.norm(data['vec'][v1_ind]-data['vec'][v2_ind])
		M[v1_ind][v2_ind] = d
		M[v2_ind][v1_ind] = d

# Data message
print('done.\nInput is {:g} points in {:g} dimensions\n'.format(len(data_raw), len(data.at[0,'vec'])) + '-'*40)

###
### Filter functions
###

def filt(v):
	return nearest(v,3) # MAKE OPTIONABLE LATER

# nearest: Returns the distance at which the input vector finds its num nearest neighbors in data. 
# Returns 0 if num > |data| .
def nearest(vec_id,num):
	if num > n:
		return 0
	else:
		return sorted(M[vec_id])[num]

# density: 

# eccentricity:

# graph_laplacian:

###
### Clustering functions
###

def clust(vectors):
	return kmeans(vectors) # MAKE OPTIONABLE LATER

# kmeans: 
def kmeans(sample):
	initial_centers = kmeans_plusplus_initializer(sample, 2).initialize()
	xmeans_instance = xmeans(sample, initial_centers, 20)
	xmeans_instance.process()
	return xmeans_instance.get_clusters()

# slc: single linkage clustering

###
### Pipeline
###

# Filter step: get filter values, range, intervals
data['filt'] = [filt(i) for i in range(n)]
f_max = max(data['filt']); f_min = min(data['filt'])
overlap_abs = args.intervals*args.overlap
interval_abs = (f_max - f_min + overlap_abs + overlap_abs*args.intervals)/args.intervals
intervals_raw = [[f_min - overlap_abs + i*(interval_abs - overlap_abs), f_min + interval_abs - overlap_abs + i*(interval_abs - overlap_abs)] for i in range(args.intervals)]
intervals = pd.DataFrame(intervals_raw, columns=['start','end'])

# For each point, find the interval(s) in which it lies
data['intervals'] = [[] for i in range(n)]
intervals['members'] = [[] for i in range(args.intervals)]
for vec_ind in range(n):
	v_filt = data['filt'][vec_ind]
	cur_ind = 0
	while v_filt > intervals['end'][cur_ind]:
		cur_ind += 1
	data['intervals'][vec_ind].append(cur_ind)
	intervals['members'][cur_ind].append(vec_ind)
	if v_filt > (intervals['end'][cur_ind] - overlap_abs) and not isclose(v_filt,f_max,abs_tol=1e-5):
		data['intervals'][vec_ind].append(cur_ind+1)
		intervals['members'][cur_ind+1].append(vec_ind)

# Clustering step: get clusters of all intervals
intervals['clusters'] = [{} for i in range(args.intervals)]
for int_index in range(args.intervals):	
	current_indices = intervals['members'][int_index].copy()
	if len(current_indices) == 1:
		intervals.at[int_index,'clusters'] = {current_indices[0]:current_indices.copy()}
	elif len(current_indices) > 1:
		current_vectors = [data['vec'][i] for i in current_indices]
		clustered_indices = clust(current_vectors)
		clustered_vectors = list(map(lambda x: [intervals['members'][int_index][i] for i in x], clustered_indices.copy()))
		for cluster in clustered_vectors:
			intervals.at[int_index,'clusters'][min(cluster)] = cluster.copy()

# For each point, find the cluster(s) in which it lies
data['clusters'] = [[] for i in range(n)]
for vec_index in range(n):
	for int_index in data['intervals'][vec_index]:
		key_list = intervals['clusters'][int_index].keys()
		for key in intervals['clusters'][int_index].keys():
			if vec_index in intervals['clusters'][int_index][key]:
				data['clusters'][vec_index].append(str(int_index) + '-' + str(key))
				break

# Simplicial complex step: create lists of simplices
simp_0 = {}; simp_1 = []; simp_2 = [];
for int_index in range(args.intervals):
	for cluster_index in intervals.at[int_index,'clusters'].keys():
		simp_0[str(int_index)+'-'+str(cluster_index)] = intervals.at[int_index,'clusters'][cluster_index]
		if int_index < args.intervals-1:
			for cluster_2index in intervals.at[int_index+1,'clusters'].keys():
				intersections = [p for p in intervals.at[int_index,'clusters'][cluster_index] if p in intervals.at[int_index+1,'clusters'][cluster_2index]] 
				if intersections != []:
					simp_1.append((str(int_index)+'-'+str(cluster_index), str(int_index+1)+'-'+str(cluster_2index)))
					if (args.overlap > .5) and (int_index < args.intervals-2):
						for cluster_3index in intervals.at[int_index+2,'clusters'].keys():
							triple = [p for p in intervals.at[int_index+2,'clusters'][cluster_3index] if p in intersections]
							if triple != []:
								simp_1.append((str(int_index)+'-'+str(cluster_index), str(int_index+1)+'-'+str(cluster_2index), str(int_index+2)+'-'+str(cluster_3index)))

# Pipeline message
print('(user) Filter function:     Nearest 3 neighbors\n(user) Number of intervals: {:g}\n(user) Percent overlap:     {:.2f}'.format(args.intervals, args.overlap*100))
print('-'*40)
print('(mapper) Range of function:   [{:.2f},{:.2f}]\n(mapper) Length of intervals: {:.2f}'.format(f_min, f_max, interval_abs))

###
### Output to desired format
###

# output=mpl: Write to matplotlib graph
max_cluster_size = max(list(map(lambda x: len(x), list(simp_0.values()))))

# input: size of some cluster (integer)
# output: number between 0 and 1 representing realtive size (float)
def setsize(val):
	return sqrt(val/max_cluster_size)

# input: interval number (integer)
# output: color representing filter value
def setcolor(val):
	return cm.rainbow(1-val/args.intervals)

if args.output == 'mpl':
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		G = nx.Graph()
		G.add_nodes_from(simp_0.keys())
		G.add_edges_from(simp_1)
		nx.draw_spring(G, 
			node_size = [400*setsize(len(simp_0[v])) for v in simp_0.keys()], 
			node_color = [setcolor(int(v.split('-')[0])) for v in simp_0.keys()],
			k=0.15,
			iterations=50)
		plt.savefig('%gint-%govl.png'%(args.intervals,round(args.overlap*100))) 

# output=txt: Write to file
if args.output == 'txt':
	f = open('sc-%gint-%govl'%(args.intervals,round(args.overlap*100)),'w')
	for s0 in simp_0.keys():
		f.write('0 ' + s0 + ' ')
		for vec in simp_0[s0]:
			f.write(data.at[vec,'id']+',')
		f.write('\n')
	for s1 in simp_1:
		f.write('1 ' + s1[0] + ' ' + s1[1] + '\n')
	for s2 in simp_2:
		f.write('2 ' + s2[0] + ' ' + s2[1] + ' ' + s2[2] + '\n')
	f.close()

# End message
print('(mapper) Number of clusters:  %g'%(len(simp_0)))
if args.output == 'mpl':
	print('(mapper) Output image file:   %gint-%govl.png'%(args.intervals,round(args.overlap*100)))
if args.output == 'txt':
	print('(mapper) Output text file:    %gint-%govl'%(args.intervals,round(args.overlap*100)))	
print('-'*40)
exit()