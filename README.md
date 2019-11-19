# easy-mapper
An easy-to-use python implementation of the mapper algorithm

+ Runs in Python 3.7
+ Input: text file with vectors on each line, coordinates separated by spaces
+ Output: image of simplicial complex of data, or text file with simplicial complex structure

## Useage
The packages `numpy`, `pandas`, `networkx`, `matplotlib`, and `pyclustering` are necessary. Default usage is:

```
python easy-mapper.py datafile
```
A `.png` file will be created in your current directory that visualizes the data.

### Basic arguments
Usage with more options is possible, with the following arguments:
+ `--filter=projection` : Type of filter function to use.
+ `--intervals=10` : Number of intervals to use.
+ `--overlap=0.1` : Overlap percentage from one interval to the next.
+ `--out='mpl'` : Type of output. Default is an image made by `matplotlib`, also `'txt'` can be used for a text file output.

### Extended arguments
+ `--filter_projection`
+ `--filter_nearest`
+ `--filter_density`
+ `--ids` : Only relevant for the `--output='txt'` option. If `--ids=True`, then the input file is assumed to have a first column that holds the name for the vector on that line. The ids are printed in the output text file, and can be useful for making finer graphs. 
+ `--out_labels`
+ `--out_legend`
+ `--out_legend_n`

### Examples
[[https://github.com/jlazovskis/easy-mapper/blob/master/examples/circle.png|alt=sampled_circle]]

### Caveats
There are several shortcomings to the `easy-mapper` implementation:
+ Only works with 1-dimensional filter functions
+ Cluster function has to be changed manually by (un)commenting code. This will be updated in later versions.
+ No way to use custom filter function except by changing the code. Goal is to use auxiliary file containing filter function as input.

## History
2019-11-19 : Added example of circle. Renamed output file. User messaged if filter is unrecognized.<br>
2019-11-17 : Filter option functional: projection, nearest neighbors, or density.<br>
2019-11-06 : Split main file into subfiles. Added "both" option for output (matplotlib and write to text file).<br>
2019-11-03 (<strong>v1</strong>) : Github repo initiated, first version with basic functionality and options<br>
2019-10-31 : Work started on easy-mapper<br>
2007: Carlsson--Singh--Memoli publish the `mapper` paper [Topological Methods](https://research.math.osu.edu/tgda/mapperPBG.pdf) on which this is based

## LOTTD
+ Make a table of why this is useful, compare with other mapper versions
+ Make more examples (swirl, 3D)
+ Make cluster function optionable
+ Add custom and 2-dimensional filter options