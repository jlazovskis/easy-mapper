###
### Output and drawing functions for easy-mapper
###

# Import packages
import networkx as nx            # creating graphs
import matplotlib.pyplot as plt  # drawing graphs
from matplotlib import cm        # coloring graphs
import warnings                  # supressing warnings
from math import sqrt            # drawing nodes at the right size

# input: size of some cluster (integer)
# output: number between 0 and 1 representing realtive size (float)
def setsize(vertices,val):
	max_cluster_size = max(list(map(lambda x: len(x), list(vertices.values()))))
	return sqrt(val/max_cluster_size)

# input: interval number (integer)
# output: color representing filter value
def setcolor(interval_number,val):
	return cm.rainbow(1-val/interval_number)

def do_output(interval_number,overlap_percent,out_type,vertices,edges,faces):
	if out_type == 'mpl' or out_type == 'both':
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			G = nx.Graph()
			G.add_nodes_from(vertices.keys())
			G.add_edges_from(edges)
			nx.draw_spring(G, 
				node_size = [400*setsize(vertices,len(vertices[v])) for v in vertices.keys()], 
				node_color = [setcolor(interval_number,int(v.split('-')[0])) for v in vertices.keys()],
				k=0.15,
				iterations=50)
			plt.savefig('%gint-%govl.png'%(interval_number,round(overlap_percent*100))) 
	if out_type == 'txt' or out_type == 'both':
		f = open('sc-%gint-%govl'%(interval_number,round(overlap_percent*100)),'w')
		for s0 in vertices.keys():
			f.write('0 ' + s0 + ' ')
			for vec in vertices[s0]:
				f.write(data.at[vec,'id']+',')
			f.write('\n')
		for s1 in edges:
			f.write('1 ' + s1[0] + ' ' + s1[1] + '\n')
		for s2 in faces:
			f.write('2 ' + s2[0] + ' ' + s2[1] + ' ' + s2[2] + '\n')
		f.close()