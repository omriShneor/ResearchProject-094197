from Utils.dataset_utils import trigger_threshold,compute_condensed_distance_matrix
from GrowthCurves.growth_curves import extract_clean_wieght_data,extract_dataframe_per_id,add_delta_from_start_in_hours
from scipy.cluster.hierarchy import dendrogram, linkage
from defenitions import ROOT_DIR
from matplotlib import pyplot as plt
import numpy as np

def create_dictionary_of_time_series(df_per_id):
    timeseries_dict = {}
    for df in df_per_id:
        ts = df.set_index('Hours_From_First_Sample')
        id = ts['ID'].values[0]
        ts = ts.drop(columns = ['ID','Date'])
        timeseries_dict[id] = ts
    return timeseries_dict

def clustering_based_distances(distance_array,timeseries_keys,method='single'):
    linked = linkage(distance_array, method)
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(linked,labels=timeseries_keys)
    directory = ROOT_DIR + "\\ClusterFigures\\"
    plt.savefig(directory+'Hierarchical-clustering-' + method)
    plt.clf()


cleaned_df = extract_clean_wieght_data()
df_per_id = extract_dataframe_per_id(cleaned_df)
df_per_id = trigger_threshold(df_per_id, 1500)
df_per_id = add_delta_from_start_in_hours(df_per_id)
timeseries_dict = create_dictionary_of_time_series(df_per_id)
timeseries_list = list(timeseries_dict.values())
timeseries_list = [series.Weight for series in timeseries_list]
timeseries_keys = list(timeseries_dict.keys())
np_array_timeseries = np.array(timeseries_list)
distance_array = compute_condensed_distance_matrix(np_array_timeseries[:100])
clustering_based_distances(distance_array,timeseries_keys[:100],'weighted')