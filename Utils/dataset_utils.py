import pandas as pd
from defenitions import ROOT_DIR

def get_dataset(dataset_filename):
    datasets_dir = ROOT_DIR + '\\DataSet'
    dataset = pd.read_excel(datasets_dir + '\\' + dataset_filename,na_values="NULL")
    return dataset