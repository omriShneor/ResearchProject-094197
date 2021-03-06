import logging
from builtins import len
from defenitions import ROOT_DIR
from Utils.dataset_utils import get_dataset,trigger_threshold,hash_df_ids
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GrowthCurves.function_fitting import polyniomial_fitting


def extract_clean_wieght_data():
    '''
    extracting the data from the dataset, logging information about the data,
    cleaning the data and returning the cleaned data.
    :return pandas.DataFrame df: the cleaned data after removing NULL values from the Weight column
    '''
    df = get_dataset("WeightData.xlsx")
    initial_rows = len(df.index)
    logging.info("Initial number of Rows: " + str(initial_rows))
    df = df.dropna(subset=["Weight"])
    cleaned_df_rows = len(df.index)
    # TODO: Need to put all the prints in a proper logger and not just print to console.
    logging.info("Cleaned number of Rows: " + str(cleaned_df_rows))
    logging.info("Number or Rows lost: " + str(initial_rows - cleaned_df_rows))
    logging.info("Percentage of Data lost:" + str((initial_rows - cleaned_df_rows)/initial_rows))
    return df


def extract_dataframe_per_id(df):
    '''
    extracting dataframe per each ID, and sorting the data by ascending time order.
    :param df: The dataframe we want to extract the data from, it should have 3 columns,
                ['ID','Weight','Date'].
    :return df: a list of pandas.DataFrame for each ID sorted by ascending order
    '''
    id_set = set(df['ID'])
    df_per_id = [df.loc[df['ID'] == uid] for uid in id_set]
    # Reverse the order of the dataframe so the dates are in ascending order.
    df_per_id = [data.iloc[::-1] for data in df_per_id]
    for df in df_per_id:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Weight'] = df['Weight'] * 1000
    return df_per_id

def add_delta_from_start_in_hours(df_per_id):
    '''
    adding a column of 'Hours from first sample` to each dataframe describing the
    time difference between each test and the first test taken for that ID.
    :param df_per_id: a list of pandas.DataFrame for each ID with
    :return df: a list of pandas.DataFrame for each ID with with a 4th column called 'Hours_From_First_Sample'
    the described column above added to it
    '''
    for df in df_per_id:
        df['Hours_From_First_Sample'] = (df['Date'] - df['Date'].iloc[0])
        df['Hours_From_First_Sample'] = df['Hours_From_First_Sample']/np.timedelta64(1,'h')
    return df_per_id

def plot_data_per_id(df_per_id,expected_weight_df):
    '''
    adding a column of 'Hours from first sample` to each dataframe describing the
    time difference between each test and the first test taken for that ID.
    :param df_per_id:
    :return df: a ist of pandas.DataFrame for each ID with
    the described column above added to it
    '''
    for i in range(len(df_per_id)):
        normalized_expected_weight_df = clean_expected_weight_df(df_per_id[i], expected_weight_df)
        (x1,y1) = polyniomial_fitting(normalized_expected_weight_df,'Hours_From_First_Sample','ExpectedWeight',8)
        # (x2,y2) = polyniomial_fitting(df_per_id[i],'Hours_From_First_Sample','Weight',8)
        plt.plot(x1,y1,label='Expected Weight',color='r')
        plt.scatter(df_per_id[i]['Hours_From_First_Sample'],df_per_id[i]['Weight'],label='Weight',color='b',s=2**2)
        directory = ROOT_DIR + "\\Plots\\"
        plt.savefig(directory+'plt-result-' + str(df_per_id[i]['ID'].real[0]) + '.png')
        plt.clf()
    return


def clean_expected_weight_df(df, expected_weight_df):
    '''
    cleaning the expected weight dataframe. the result is a dataframe with 2 columns: Hours_From_First_Sample,
    ExpectedWeight.
    :param df: a dataframe representing the weight overtime of a single subject.
    :param expected_weight_df: a dataframe that holds the data regarding the expected weight values over time
    for all the subjects.
    :return suitable_df: the suitable expected dataframe weight over time for the df input parameter.
    '''
    idx = filter_suitable_expected_weight(df, weight_at_birth_series)
    suitable_df = expected_weight_df.iloc[:, [0, idx]]
    suitable_df.iloc[:, 0] = suitable_df.iloc[:, 0] * 24
    suitable_df = suitable_df.dropna()
    suitable_df.columns = range(suitable_df.shape[1])
    suitable_df.rename(columns={0: 'Hours_From_First_Sample', 1: 'ExpectedWeight'}, inplace=True)
    return suitable_df


def collect_expected_weight():
    '''
    collecting the expected weight dataframe from an excel file called: "ExpectedWeight.xlsx" which
    needs to be in the path ROOT_DIR\DataSet\ExpectedWeight.xlsx
    :return df: the dataframe
    '''
    # collect the dataset and remove the headers from it.
    datasets_dir = ROOT_DIR + '\\DataSet'
    df = pd.read_excel(datasets_dir + '\\' + "ExpectedWeight.xlsx", header=None,skiprows=1)
    return df

def collect_weight_per_date_of_birth(df):
    '''
    Remove the column for Days old e.g. the zero redundent value
    :return df: the dataframe
    '''
    return df.iloc[0][1::]

def filter_suitable_expected_weight(df_per_id,weight_at_birth_series):
    '''
    use binary search for getting the suitable expected weight column of the dataframe of weights sent as input.
    :param df_per_id: the df per id we want to find the suitable expected weight for.
    :param weight_at_birth_series: a series of values of weight at birth, from which we extract the idx of
    the expected weight column index.
    :return df: the idx in the expected weight data frames of the suitable column for the given df as input.
    '''
    weight_at_birth_in_grams = df_per_id.iloc[0]["Weight"]
    idx = weight_at_birth_series.searchsorted(weight_at_birth_in_grams, side='left')
    if idx[0] == len(weight_at_birth_series):
        return idx[0]
    return idx[0]+1

def plot_data_per_id_with_normalized_time_from_birth(df_per_id,expected_weight_df):
    '''
    add delta from start column to the df per id and plot the data per id.
    :param df_per_id: the df we want to add normalized plot and plot.
    :param expected_weight_df: the expected weight we want to plot.
    '''
    df_per_id = add_delta_from_start_in_hours(df_per_id)
    plot_data_per_id(df_per_id,expected_weight_df)


# Data Preprocessing phase!
logging.basicConfig(filename="GrowthCurves.log", level=logging.INFO)
cleaned_df = extract_clean_wieght_data()
expected_weight_df = collect_expected_weight()
weight_at_birth_series = collect_weight_per_date_of_birth(expected_weight_df)
df_per_id = extract_dataframe_per_id(cleaned_df)
df_per_id = add_delta_from_start_in_hours(df_per_id)
df_per_id = [df.drop(columns = ['Date']) for df in df_per_id]
df_per_id = trigger_threshold(df_per_id, 1500)
df_per_id = hash_df_ids(df_per_id)

# Plotting the data!
plot_data_per_id(df_per_id,expected_weight_df)
