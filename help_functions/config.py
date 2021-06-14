from pathlib import Path


# Define parent path
parent_path=str(Path.cwd().parent)


# Define data folder paths
data_path = parent_path + '\\data'

cleaned_data_path = data_path + '\\cleaned\\'
processed_data_path = data_path + '\\processed\\'
raw_data_path = data_path + '\\raw\\'

