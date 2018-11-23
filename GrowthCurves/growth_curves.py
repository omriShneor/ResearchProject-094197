import logging

from Utils.dataset_utils import get_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        df['Date'] = pd.to_datetime(df['Date'])
        df['Hours_From_First_Sample'] = (df['Date'] - df['Date'].iloc[0])
        df['Hours_From_First_Sample'] = df['Hours_From_First_Sample']/np.timedelta64(1,'h')
    return df_per_id

def plot_data_per_id(df_per_id,plot_kind):
    '''
    adding a column of 'Hours from first sample` to each dataframe describing the
    time difference between each test and the first test taken for that ID.
    :param df_per_id:
    :return df: a ist of pandas.DataFrame for each ID with
    the described column above added to it
    '''
    for i in range(len(df_per_id)):
        df_per_id[i].plot(kind=plot_kind,x='Hours_From_First_Sample',y='Weight',color='red',
                yticks=np.arange(0, 6, 0.2),ylim = (0,6))
        directory = "C:\\Users\\omrish\\Desktop\\School\\ProejctWithOfra\\ProjectSourceCode\\Plots\\"
        plt.savefig(directory+'plt-result-' + str(df_per_id[i]['ID'].real[0]) + '.png')
        plt.clf()
    return


logging.basicConfig(filename="GrowthCurves.log", level=logging.INFO)
cleaned_df = extract_clean_wieght_data()
df_per_id = extract_dataframe_per_id(cleaned_df)
df_per_id = add_delta_from_start_in_hours(df_per_id)
plot_data_per_id(df_per_id,'scatter')