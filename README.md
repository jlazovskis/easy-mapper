# easy-mapper
An easy-to-use python implementation of the mapper algorithm

+ Runs in Python 3.7
+ Input: text file with vectors on each line, coordinates separated by spaces
+ Output: image of simplicial complex of data, or text file with simplicial complex structure

## Useage
The packages `numpy`, `pandas`, `networkx`, `matplotlib` are necessary. Default usage is:

```
python easy-mapper.py datafile
```
A `.png` file will be created in your current directory that visualizes the data.

### Optional arguments
Usage with more options is possible, with the following arguments:
+ `--intervals=10` : Number of intervals to use.
+ `--overlap=0.1` : Overlap percentage from one interval to the next.
+ `--output='mpl'` : Type of output. Default is an image made by `matplotlib`, also `'txt'` can be used for a text file output.
+ `--ids=False` : Only relevant for the `--output='txt'` option. If `--ids=True`, then the input file is assumed to have a first column that holds the name for the vector on that line. The ids are printed in the output text file, and can be useful for making finer graphs. 

Example usage with these arguments is:
```
python easy-mapper.py datafile --intervals=4 --overlap=0.5 --output='txt' --ids=True
```

### Caveats
There are several shortcomings to the `easy-mapper` implementation:
+ Only works with 1-dimensional filter functions
+ Cluster and filter functions have to be changed manually by (un)commenting code. This will be updated in later versions.

## History
2019-11-06 : Split main file into subfiles. Added "both" option for output (matplotlib and write to text file).<br>
2019-11-03 (<strong>v1</strong>) : Github repo initiated, first version with basic functionality and options<br>
2019-10-31 : Work started on easy-mapper<br>
2007: Carlsson--Singh--Memoli publish the `mapper` paper [Topological Methods](https://research.math.osu.edu/tgda/mapperPBG.pdf) on which this is based

## LOTTD
+ Make a table of why this is useful, compare with other mapper versions
+ Make examples (circle, swirl)
+ Make filter and cluster funnctions optionable
+ Add 2-dimensional filter options