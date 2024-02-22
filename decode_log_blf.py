import os
import sys
import csv
import re

from io import StringIO

original_stderr = sys.stderr
sys.stderr = StringIO()

import cal
import cantool
import multiprocessing
import time as tm
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from fnmatch import fnmatch

import parallel_cores as pc
from util import RunTimer

sys.stderr = original_stderr

def get_all_logs_paths(logs_dir):
    logs_path_selected = []
    for dir_path, subdir, files in os.walk(logs_dir):
        logs_path_selected.append(os.join(dir_path, files))
    return logs_path_selected

def get_all_dbc_path(dbc_dir):
    dbc_path_selected = []
    for index, (dir_path, subdir, files) in enumerate(os.walk(dbc_dir)):
        dbc_path_selected.append((os.join(dir_path, files), index+1))
    return dbc_path_selected

def is_numeric(value):
    try:
        float(value)
    except:
        ValueError
        return False
    return True

def get_message_id_list(db):
    message_id_list = {}
    for message in db._messages:
        if int(message.frame_id) not in message_id_list:
            message_id_list[int(message.frame_id)] = True

    return message_id_list

def get_message_name_list(signal_list_path):
    message_name_list = {}

    with open(signal_list_path, 'r') as file_in:
        csv_file = csv.reader(file_in)
        csv_list = list(csv_file)
        for signal in csv_list:
            if signal[0] not in message_name_list:
                message_name_list[signal[0]] = True

    return message_name_list

def decode_message(db, message):
    decoded_message = db.decode_message(message.arbitration_id, 
                                        message.data,
                                        allow_truncated = True)
    return decoded_message

def is_message_id_in_list(message, message_id_list):
    return message.arbitration_id in message_id_list

def is_signal_in_message_name_list(signal, message_name_list):
    return signal in message_name_list

def clear_float_nan(df):
    for signal in df.keys():
        try:
            # GEt thefist non-NAN vlaue in column 'A'
            first_valid_index = df[signal].first_valid_index()
            a = float(df[signal].iloc[first_valid_index])
        except:
            df[signal] = df[signal].fillna('NaN')

    return df

def is_dir_exist(dir, parent):
    b_dir_exist = os.path.exists(dir)
    if not b_dir_exist:
        dir = os.path.join(parent, dir)
        b_dir_exist = os.path.exists(dir)

        if b_dir_exist:
            return dir
        else:
            raise ValueError(f"Cannot find the folder {dir}")
    else:
        return os.path.join(os.getcwd(), dir)
    
def gen_parent_dir(dir, dir_gen):
    parent_dir = os.path.dirname(dir)
    dir_gen = os.path.join(parent_dir, dir_gen)
    if not os.path.exists(dir_gen):
        os.makedirs(dir_gen)
    return dir_gen

def dic_default_value(keys, default_value = float('nan')):
    my_dict = {key: default_value for key in keys}
    return my_dict