import pandas as pd

def get_dataset(dataset_filename):
    datasets_dir = 'C:\\Users\\omrish\\Desktop\\School\\ProejctWithOfra\\ProjectSourceCode\\DataSet'
    dataset = pd.read_excel(datasets_dir + '\\' + dataset_filename)
    return dataset