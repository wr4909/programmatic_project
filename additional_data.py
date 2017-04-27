import pandas as pd
import os
import numpy as np

"""
Loads all csvs into a list of DataFrames
Returns that list and a dictionary that can be used to call find each DataFrame by name.

Sample Use:
df_list, index_dict = load_all_csv()
df_list[index_dict["ip"]]
"""

def load_all_csv(directory="/additional_data"):
    cur_dir = os.getcwd()
    target_directory = cur_dir + directory + "/"
    files = [ f for f in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory,f)) ]
    dfs = {file_csv: pd.read_csv(target_directory + file_csv, encoding='latin1') for file_csv in files}
    def rowDict(f):
        keys = f[f.columns.values.tolist()[0]].tolist()
        f = f[f.columns.values.tolist()[1:]]
        valArray = f.ix[:, f.dtypes == float].as_matrix()
        return {k: valArray[i] for i,k in enumerate(keys)}, valArray.shape[1]

    return {k: rowDict(v) for k,v in dfs.items()}

"""
Takes a normal DataFrame from part 1 (or a slice) and inner joins using the additional
DataFrame, on a specified column from each.
"""

def inner_join(normal_df, additional_pair, normal_df_column_name):
    additional, width = additional_pair
    data = []
    for d in normal_df[normal_df_column_name]:
        if d in additional:
            data.append(additional[d])
        else:
            data.append(np.zeros(width))

    return np.stack(data)

if __name__ == "__main__":
    pass
