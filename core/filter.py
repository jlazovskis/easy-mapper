###
### Filter functions for easy-mapper
###

## From Carlsson-Singh-Memoli paper

from math import exp

# density: Uses a Gaussian kernel, epsilon must be user-defined. Large epsilon means smooth sample.
def density(distance_matrix,vec_id,normalizer,epsilon):
	return normalizer*sum(list(map(lambda x: exp(-(x**2)/epsilon), distance_matrix[vec_id])))

# eccentricity:

# graph_laplacian:

## Other classic filter functions

# projection: Projects to a given axis
def projection(data_panda,vec_id,axis):
	return data_panda.at[vec_id,'vec'][axis]

# nearest: Returns the distance at which the input vector finds its num nearest neighbors in data. 
# Returns 0 if num > |data| .
def nearest(distance_matrix,vec_id,num):
	if num > len(distance_matrix):
		return 0
	else:
		return sorted(distance_matrix[vec_id])[num]