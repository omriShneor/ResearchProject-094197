import pandas as pd
from defenitions import ROOT_DIR
from math import fabs
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler


def get_dataset(dataset_filename):
    datasets_dir = ROOT_DIR + '\\DataSet'
    dataset = pd.read_excel(datasets_dir + '\\' + dataset_filename,na_values="NULL")
    return dataset

def costum_euclide_norm(ts1,ts2):
    return fabs(norm(ts1)-norm(ts2))

def normalize_timeseries_values(ts):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(ts)
    # print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    return scaler.transform(ts)