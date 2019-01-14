import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This this is the project root dir.
TIME_SERIES_ARRAY_SIZE = 196  # This is the array size of the number of weight plots used for clustering.
METHOD = 'average' # hierarchical cluster distance method used to determine how to cluster the nodes.
DEPTH = 10 # Depth of cluster level to display the plots in.
