import itertools
import pandas as pd
from defenitions import ROOT_DIR
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sys import maxsize
from scipy import stats


def get_dataset(dataset_filename):
    '''
    collecting the dataframe from an excel file with the name dataset_filename which
    needs to be in the path ROOT_DIR\DataSet\
    :return df: the dataframe requested.
    '''
    datasets_dir = ROOT_DIR + '\\DataSet'
    dataset = pd.read_excel(datasets_dir + '\\' + dataset_filename,na_values="NULL")
    return dataset

def dtw_metric(s1,s2):
    '''
    computing FASTDtw metric of two timeseries. s1,s2 and returning the result.
    :param s1: timeseries 1.
    :param s2: timeseries 2.
    :return: returning the FastDTW metric of the two timeseries.
    '''
    distance, path = fastdtw(s1, s2, dist=euclidean)
    return distance

def normalize_timeseries_values(ts):
    '''
    normalizing the values of a time series, to be in the range (0,1)
    :param sts: timeseries to normalize its values
    :return: returning a normalized timeseries with values between (0,1)
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(ts)
    # print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    return scaler.transform(ts)

def compute_condensed_distance_matrix(np_array_timeseries):
    '''
    implementation of scipy.spatial.distance.pdist method computing the condensed distance matrix for the given
    array of timeseries.
    :param np_array_timeseries: the array of timeseries we want to compute the condensed distance matrix for.
    :return: condensed distance matrix(upper triangular part of the square distance
    matrix strung togther into a 1D array).
    '''
    combinations = list(itertools.combinations(range(len(np_array_timeseries)), 2))
    distances = [dtw_metric(np_array_timeseries[combinations[i][0]],np_array_timeseries[combinations[i][1]]) for
                 i in range(len(combinations))]
    distances = np.array(distances)
    return distances


def trigger_threshold(df_per_id,threshold):
    '''
    remove all the dataframes of all the subjects who were born with weight over the threshold grams.
    :param df_per_id: the df array we want to trigger the threshold on.
    :param threshold: the threshold in grams of the weight in which the subjects were born we want to examn.
    :return: a list of dataframes with only subjects who were born with weight under the threshold.
    '''
    df_per_id = [df for df in df_per_id if df['Weight'].values[0] < threshold]
    df_per_id = [df[~(np.abs(df.Weight-df.Weight.mean()) > (3*df.Weight.std()))] for df in df_per_id]
    df_per_id = [df[~(np.abs(df.Hours_From_First_Sample - df.Hours_From_First_Sample.mean()) >
                      (3 * df.Hours_From_First_Sample.std()))] for df in df_per_id]
    return df_per_id


def hash_df_ids(df_per_id):
    '''
    hashing the column ID in every data frame.
    :param df_per_id: a list of data frames we want to hash the IDs of.
    :return: return the updated list of data frames.
    '''
    for df in df_per_id:
        df['ID'] = df['ID'].apply(lambda x: hash(x)+maxsize+1)
    return df_per_id
