###
### Output and drawing functions for easy-mapper
###

# Import packages
import numpy as np               # storing and working with vectors
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

def do_output(interval_number,overlap_percent,out_type,out_legend,out_legend_n,data_panda,vertices,edges,faces):
	if out_type == 'mpl' or out_type == 'both':
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			if out_legend:
				spacing_factor = .06
				plt.figure(figsize=(6,8+spacing_factor*len(vertices.keys())))
			G = nx.Graph()
			G.add_nodes_from(vertices.keys())
			G.add_edges_from(edges)
			pos=nx.spring_layout(G) 
			nx.draw(G,pos,
				node_size = [400*setsize(vertices,len(vertices[v])) for v in vertices.keys()], 
				node_color = [setcolor(interval_number,int(v.split('-')[0])) for v in vertices.keys()],
				k=0.15,
				iterations=50)
			labels = {lab : lab for lab in vertices.keys()}
			label_pos = {lab : pos[lab] + np.array([.065,.065]) for lab in pos.keys()}
			if out_legend:
				label_pos = {lab : pos[lab] + np.array([.09,.09]) for lab in pos.keys()}
				x_values, y_values = zip(*pos.values()); x_min = min(x_values); x_max = max(x_values); y_min = min(y_values); y_max = max(y_values)
				legend = sorted([[len(vertices[v]),v] for v in vertices.keys()], key=lambda x: x[0])
				legend.reverse()
				y_min += -0.25
				for v_ind in range(len(vertices)):
					plt.text(x_min,y_min,legend[v_ind][1]+': ', ha='right',va='bottom',fontsize=8,alpha=.6,fontweight='bold')
					plt.text(x_min+.05,y_min,'('+str(legend[v_ind][0])+')', ha='center',va='bottom',fontsize=8,alpha=.6)
					if legend[v_ind][0] <= out_legend_n:
						plt.text(x_min+.12,y_min,str(vertices[legend[v_ind][1]])[1:-1], ha='left',va='bottom',fontsize=8,alpha=.6)
					else:
						plt.text(x_min+.12,y_min,str(vertices[legend[v_ind][1]][:out_legend_n])[1:-1]+'...', ha='left',va='bottom',fontsize=8,alpha=.6)  # first 3 labels
#						plt.text(x_min+.12,y_min,str(vertices[legend[v_ind][1]])[1:-1], ha='left',va='bottom',fontsize=8,alpha=.6)            # all labels
					y_min += -spacing_factor
				plt.xlim(x_min - (x_max-x_min)*.1, x_max + (x_max-x_min)*.1)
				plt.ylim(y_min, y_max + (y_max-y_min)*.1)
			label_bg = dict(boxstyle="round", ec="white", fc="white", alpha=0.9)
			nx.draw_networkx_labels(G,label_pos,labels,bbox=label_bg,font_size=8,alpha=.6)
			plt.savefig('%gint-%govl.png'%(interval_number,round(overlap_percent*100)),dpi=300) 
	if out_type == 'txt' or out_type == 'both':
		f = open('sc-%gint-%govl'%(interval_number,round(overlap_percent*100)),'w')
		for s0 in vertices.keys():
			f.write('0 ' + s0 + ' ')
			for vec in vertices[s0]:
				f.write(data_panda.at[vec,'id']+',')
			f.write('\n')
		for s1 in edges:
			f.write('1 ' + s1[0] + ' ' + s1[1] + '\n')
		for s2 in faces:
			f.write('2 ' + s2[0] + ' ' + s2[1] + ' ' + s2[2] + '\n')
		f.close()