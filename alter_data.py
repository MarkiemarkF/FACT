import os
import sys
import csv

import numpy as np
import pandas as pd


def check_data(data_path: str) -> None:
    """
    Checks if data exists, exits otherwise.

    :param data_path: path to data
    """
    if not os.path.exists(data_path):
        print(data_path)
        print('CANNOT FIND DATA TO LOAD!')
        sys.exit(1)


def convert_data_columns(load_path: str, store_path: str, col_num: int, convs: list) -> None:
    """
    Converts values in a column of a data file (e.g. gender 'male' &'female' to 0 & 1).
    Saves result in store_path location.

    :param load_path: path to data to alter
    :param store_path: path to store altered data
    :param col_num: indicates column that has to be altered
    :param convs: conversions, list of tuples, [(CURRENT_VALUE_1, NEW_VALUE_1), (CURRENT_VALUE_2, NEW_VALUE_2), ...]
    """
    # Create the correct data_paths
    load_path = os.path.join(os.getcwd(), 'data', load_path)
    store_path = os.path.join(os.getcwd(), 'data', store_path)

    # Check if datafile exists
    check_data(load_path)

    # Iterate over file to apply all conversions
    file = csv.reader(open(load_path))

    lines = list(file)

    for line in lines[1:]:
        attr = line[col_num]
        changed_flag = 0

        # Iterate over all conversion
        for conv in convs:
            if attr == conv[0]:
                line[col_num] = conv[1]
                changed_flag = 1

        # Mention if conversion has not been performed (e.g. due to typo, or wrong column number)
        if changed_flag == 0:
            print('Nothing converted on line:', line)
            print(f'On attribute: {attr}\n')

    # Save the altered datafile to the store_path
    writer = csv.writer(open(store_path, 'w', newline=''))
    writer.writerows(lines)
    print('Job done, wrote data to:', store_path)


def create_subset(data_path: str, store_name: str, n: int, sep: str=',') -> None:
    """
    Creates a subset of length n of a dataset.
    Saves result in 'data/{store_name}_{n}' location.

    :param data_path: path to data to create a subset from
    :param store_name: filename to store subset of data
    :param n: number of samples in subset
    :param sep: separator used in datafile
    """
    # Check if datafile exists
    data_path = load_path = os.path.join(os.getcwd(), 'data', data_path)
    check_data(load_path)

    # Create subset of data
    df = pd.read_csv(data_path, sep=sep)
    df_sub = df.sample(n)

    # Store data subset
    store_path = os.path.join(os.getcwd(), 'data', store_name.title(), store_name + '_' + str(n) + '.csv')
    df_sub.to_csv(store_path, sep=sep)

    print('Job done, wrote data to:', store_path)


def remove_first_column(load_path: str, store_path: str) -> None:
    """
    Removes the first column of a datafile and stores the modded data in a specified location (used for ID removal).
    Saves the altered data in store_path location

    :param load_path: path to data to remove the first column from
    :param store_path: path to store the altered data
    """
    # Check if data exists & read file
    check_data(load_path)
    file = csv.reader(open(load_path))
    lines = list(file)

    # Write file to store_path without the first column
    new_lines = [line[1:] for line in lines]
    writer = csv.writer(open(store_path, 'w', newline=''))
    writer.writerows(new_lines)
    print('Job done, wrote data to:', store_path)


def npz_to_csv(load_path: str, store_path: str) -> None:
    """
    Converts .npz file from load_path to .csv at store_path

    :param load_path: path to npz file to convert
    :param store_path: path to store the data
    """
    data = np.load(load_path)
    pd.DataFrame(data).to_csv(store_path)


if __name__ == '__main__':
    # Comment out unwanted functions
    convert_data_columns(os.path.join('Student', 'student_mat_Cortez.csv'), 'Student/student_mat_Cortez_sexmod.csv', 1,
                         [('M', 0), ('F', 1)])
    #
    # create_subset(os.path.join('Bank', 'bank-additional-full.csv'), 'bank', 5, sep=';')
    #
    # remove_first_column(os.path.join(os.getcwd(), 'data', 'German_Credit', 'german_credit_data.csv'),
    #                     os.path.join(os.getcwd(), 'data', 'German_Credit', 'german_credit_mod2.csv'))
    #
    # remove_first_column(os.path.join(os.getcwd(), 'data', 'Drugnet', 'DRUGNET_mod.csv'),
    #                     os.path.join(os.getcwd(), 'data', 'Drugnet', 'DRUGNET_mod.csv'))
    #
    # npz_to_csv(os.path.join(os.getcwd(), 'data', 'Synthetic', 'Synthetic.npz'),
    #            os.path.join(os.getcwd(), 'data', 'Synthetic', 'synthetic.csv'))
