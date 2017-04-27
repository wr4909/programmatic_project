import pandas as pd
import os

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
	df_list = []
	for file_csv in files:
		df_list.append(pd.read_csv(target_directory + file_csv))
	index_dict = {}
	for indx, file_csv in enumerate(files):
		index_dict[file_csv] = indx

	return df_list, index_dict

"""
Takes a normal DataFrame from part 1 (or a slice) and inner joins using the additional
DataFrame, on a specified column from each.
"""

def inner_join(normal_df, additional_df, normal_df_column_name, additional_df_column_name):
	return pd.merge(normal_df, additional_df, on =[normal_df_column_name, additional_df_column_name])

if __name__ == "__main__":
	pass