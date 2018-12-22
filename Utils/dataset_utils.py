import itertools
import pandas as pd
from defenitions import ROOT_DIR
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def get_dataset(dataset_filename):
    datasets_dir = ROOT_DIR + '\\DataSet'
    dataset = pd.read_excel(datasets_dir + '\\' + dataset_filename,na_values="NULL")
    return dataset

def dtw_metric(s1,s2):
    distance, path = fastdtw(s1, s2, dist=euclidean)
    return distance

def normalize_timeseries_values(ts):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(ts)
    # print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    return scaler.transform(ts)

def compute_condensed_distance_matrix(np_array_timeseries):
    combinations = list(itertools.combinations(range(len(np_array_timeseries)), 2))
    distances = [dtw_metric(np_array_timeseries[combinations[i][0]],np_array_timeseries[combinations[i][1]]) for
                 i in range(len(combinations))]
    distances = np.array(distances)
    return distances


def trigger_threshold(df_per_id,threshold):
    df_per_id = [df for df in df_per_id if df['Weight'].values[0] < threshold]
    return df_per_id
