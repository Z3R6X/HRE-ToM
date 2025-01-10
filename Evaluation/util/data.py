import json
import os
from datasets import Dataset


def load_and_prepare_data(json_file, format_function):

    # Convert dict to dataset and apply task-specific fformat function
    dict_list = json.load(open(json_file))
    ds = transform_list_to_dataset(dict_list)
    ds = ds.map(format_function, batched = True,)

    return ds


def transform_list_to_dataset(list_of_dicts):
    
    column_names = list_of_dicts[0].keys()
    dict_of_lists = dict()

    # Convert list of dicts to dict with lists
    for column_name in column_names:
        new_list = [single_dict[column_name] for single_dict in list_of_dicts]
        dict_of_lists[column_name] = new_list

    # Create dataset from dict 
    ds = Dataset.from_dict(dict_of_lists)

    return ds 


def create_directory(new_dir_path):
    try:
        os.makedirs(new_dir_path, exist_ok=True)
        print(f"Directory '{new_dir_path}' created successfully")
    except OSError as e:
        print(f"Error creating directory: {e}")