import pandas as pd

def get_dataset(dataset_filename,sheet_name):
    datasets_dir = 'C:\\Users\\omrish\\Desktop\\School\\ProejctWithOfra\\ProjectSourceCode\\DataSet'
    dataset = pd.read_excel(datasets_dir + '\\' + dataset_filename, sheet_name=sheet_name,names=["ID","Weight"])
    return dataset