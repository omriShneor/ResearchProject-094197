from Utils.dataset_utils import trigger_threshold,compute_condensed_distance_matrix,normalize_timeseries_values
from GrowthCurves.growth_curves import extract_clean_wieght_data,extract_dataframe_per_id,add_delta_from_start_in_hours
from scipy.cluster.hierarchy import dendrogram, linkage,to_tree,cut_tree,leaves_list
from defenitions import ROOT_DIR,TIME_SERIES_ARRAY_SIZE,METHOD
from functools import reduce
from matplotlib import pyplot as plt
from shutil import copy
from os import mkdir
from os.path import exists
import json
import numpy as np


def create_dictionary_of_time_series(df_per_id):
    '''
    create a dictionary from the list of df_per_id where the key is the ID of the dataframe and the value is
    a timeseries containing an index column of 'Hours_From_First_Sample' and another column of 'Weight'.
    :param df_per_id: a list of dataframes we want to convert to a dictionary.
    :return: dictionary of timeseries with IDs as keys.
    '''
    timeseries_dict = {}
    for df in df_per_id:
        ts = df.set_index('Hours_From_First_Sample')
        id = ts['ID'].values[0]
        ts = ts.drop(columns = ['ID','Date'])
        timeseries_dict[id] = ts
    return timeseries_dict

def add_node(node, parent):
    '''
    Create a nested dictionary from the ClusterNode's returned by SciPy
    :param node:
    :param parent:
    :return:
    '''
    # First create the new node and append it to its parent's children
    newNode = dict(node_id=node.id, children=[])
    parent["children"].append( newNode )
    # Recursively add the current node's children
    if node.left: add_node( node.left,newNode)
    if node.right: add_node( node.right,newNode)



def label_tree(node,dn):
    '''
    # Label each node with the names of each leaf in its subtree
    :param node: scipy.ClusterNode object, the root node of the dendrogram tree.
    :param dn: dendrogram return object.
    :return: returns the leaf labels from the dendrogram.
    '''
    if len(node["children"]) == 0:
        leafNames = [id2name(dn,node["node_id"])]
    else:
        leafNames = reduce(lambda ls, c: ls + label_tree(c,dn), node["children"], [])
    del node["node_id"]
    node["name"] = name = "-".join(sorted(map(str, leafNames)))
    return leafNames

def id2name(dn,id):
    '''
    converts an node id of a leaf in the Cluster Tree to its label in the dendrogram
    :param dn: dendrogram object.
    :param id: the node id we are looking
    :return: returns the node label.
    '''
    idx = dn['leaves'].index(id)
    return dn['ivl'][idx]

def hierarchical_cluster_to_d3_dendro(Tree,dn,method):
    d3_dendro = dict(children=[], name="Root1")
    add_node(Tree, d3_dendro)
    label_tree(d3_dendro["children"][0], dn)
    directory = ROOT_DIR + "\\ClusterFigures\\"
    json.dump(d3_dendro, open(directory + "d3-dendrogram-" + method + ".json", "w"), sort_keys=True, indent=4)
    return d3_dendro

def clustering_based_distances(distance_array,timeseries_keys,method='single'):
    '''
    Running hierarchical/agglomerative clustering using the method given as parameter,
    with the distance_array used aswell as parameter, when plotting the dendogram we use the timeseries keys as labels.
    :param distance_array: condensed distance matrix for the scipy agglomerative clustering api.
    :param timeseries_keys: the keys are used for the labels in the dendogram plot.
    :param method: the method to use for the hierarchical clustering e.g.: single,complete,average,weighted.
    '''
    Z = linkage(distance_array, method)
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z,labels=timeseries_keys)
    directory = ROOT_DIR + "\\ClusterFigures\\"
    # plt.savefig(directory+'Hierarchical-clustering-' + method)
    # plt.clf()
    T = to_tree(Z,rd=False)
    return Z,T,dn

def json_to_weight_plots(json_dict,depth,dirpath):
    if(depth == 0):
        if json_dict['name']=='Root1':
            return
        else:
            ids = json_dict['name'].split('-')
            for id in ids:
                if not exists(dirpath):
                    mkdir(dirpath)
                copy(ROOT_DIR+'\\Plots\\'+'plt-result-'+str(id)+'.png',dirpath)
            return
    for i in range(len(json_dict['children'])):
        json_to_weight_plots(json_dict['children'][i],depth-1,dirpath+ '-' +str(i))

# Data Preprocessing phase!
cleaned_df = extract_clean_wieght_data()
df_per_id = extract_dataframe_per_id(cleaned_df)
df_per_id = trigger_threshold(df_per_id, 1500)
df_per_id = add_delta_from_start_in_hours(df_per_id)
timeseries_dict = create_dictionary_of_time_series(df_per_id)
timeseries_list = list(timeseries_dict.values())
timeseries_list = [normalize_timeseries_values(series) for series in timeseries_list]
timeseries_keys = list(timeseries_dict.keys())
np_array_timeseries = np.array(timeseries_list)

# Computing the condensed distance matrix using FastDTW metric.
distance_array = compute_condensed_distance_matrix(np_array_timeseries[:TIME_SERIES_ARRAY_SIZE])

# Running hierarchical/agglomerative clustering using scipy API and ploting the dendrogram.
Z,T,dn = clustering_based_distances(distance_array, timeseries_keys[:TIME_SERIES_ARRAY_SIZE], METHOD)
d3_dendro = hierarchical_cluster_to_d3_dendro(T, dn,METHOD)

#Convert json cluster representation to dictionary
directory = ROOT_DIR + "\\ClusterFigures\\"
json_file = directory+ "d3-dendrogram-" + METHOD + ".json"
depth = 4
with open(json_file) as handle:
    json_dict = json.loads(handle.read())
    json_to_weight_plots(json_dict,depth,ROOT_DIR + "\\ClusterFigures\\"+ METHOD + "\\" + "Root")