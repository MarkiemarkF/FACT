### CODE OF THE ORIGINAL AUTHORS ###
import numpy as np
import os
from sklearn.datasets import make_blobs
import sys
import requests, zipfile, io
import pandas

__datasets = ['Adult', 'Bank', 'Synthetic', 'Synthetic-unequal', 'CensusII']

### END CODE OF THE ORIGINAL AUTHORS ###
### OUR OWN CODE ###
# Own/Custom datasets
__datasets += ['Student', 'Drugnet', 'German_Credit', 'bank_red', 'bank_2500', 'bank_5000', 'bank_10000', 'bank_15000']


def check_and_read(data_path, sep=',', header='infer'):
    """
    Check if a dataset exists and read if so.

    :param data_path: path to dataset (csv_file)
    :param sep: seperator in datafile (e.g. ',', ';', ' ')
    :param header: row number(s) to use as column names
    :return: Pandas DataFrame & dataset length
    """
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(data_path)
        print('Data does not exist. Quitting.')
        sys.exit()

    # Read dataset and return together its length
    df = pandas.read_csv(data_path, sep=sep, header=header)
    n = df.shape[0]

    return df, n
### END OF OUR OWN CODE ###


### CODE OF THE ORIGINAL AUTHORS ###
def dataset_names():
    return __datasets


def read_dataset(name, data_dir):
    data = []
    sex_num = []
    K = []
    if name not in __datasets:
        raise KeyError("Dataset not implemented:", name)

    elif name == 'Synthetic':

        n_samples = 400

        centers = [(1, 1), (2.1, 1), (1, 5), (2.1, 5)]
        data, sex_num = make_blobs(n_samples=n_samples, n_features=2, cluster_std=0.1,
                                   centers=centers, shuffle=False, random_state=1)

        index = n_samples // 2
        sex_num[0:index] = 0
        sex_num[index:n_samples] = 1
        K = 2

    elif name == 'Synthetic-unequal':

        n_samples = 400

        sample_list = [150, 150, 50, 50]
        centers = [(1, 1), (2.1, 1), (1, 3.5), (2.1, 3.5)]
        data, sex_num = make_blobs(n_samples=sample_list, n_features=2, cluster_std=0.13,
                                   centers=centers, shuffle=False, random_state=1)

        index = sample_list[0] + sample_list[1]
        sex_num[0:index] = 0
        sex_num[index:] = 1
        K = 2

    elif name == 'Adult':

        _path = 'adult.data'
        data_path = os.path.join(data_dir, _path)
        race_is_sensitive_attribute = 0

        if race_is_sensitive_attribute == 1:
            m = 5
        else:
            m = 2
        # n = 25000
        K = 10
        if (not os.path.exists(data_path)):
            print('Adult data set does not exist in current folder --- Have to download it')
            r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                             allow_redirects=True)
            if r.status_code == requests.codes.ok:
                print('Download successful')
            else:
                print('Could not download Adult data set - please download it manually')
                sys.exit()
            open(data_path, 'wb').write(r.content)

        df = pandas.read_csv(data_path, sep=',', header=None)
        # df = df[:n]
        n = df.shape[0]

        sens_attr = 9
        sex = df[sens_attr]
        sens_attributes = list(set(sex.astype(str).values))  # =[' Male', ' Female']
        df = df.drop(columns=[sens_attr])
        sex_num = np.zeros(n, dtype=int)
        sex_num[sex.astype(str).values == sens_attributes[1]] = 1

        # dropping non-numerical features and normalizing data
        cont_types = np.where(df.dtypes == 'int64')[0]  # =[0,2,4,9,10,11]
        df = df.iloc[:, cont_types]
        data = np.array(df.values, dtype=float)

        data = data[:, [0, 1, 2, 3, 5]]

        # Scale data
        # data = scale(data, axis = 0)

    elif name == 'Bank':
        # n= 6000
        # K = 4
        K = 2
        _path = 'bank-additional-full.csv'  # Big dataset with 41108 samples
        # _path = 'bank.csv' # most approaches use this small version with 4521 samples
        data_path = os.path.join(data_dir, _path)
        if (not os.path.exists(data_path)):

            print('Bank dataset does not exist in current folder --- Have to download it')
            r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip',
                             allow_redirects=True)
            # r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip', allow_redirects=True)
            if r.status_code == requests.codes.ok:
                print('Download successful')
            else:
                print('Could not download - please download')
                sys.exit()

            z = zipfile.ZipFile(io.BytesIO(r.content))
            # z.extract('bank-additional/bank-additional-full.csv','./data')
            open(data_path, 'wb').write(z.read('bank-additional/bank-additional-full.csv'))
            # open(data_path, 'wb').write(z.read('bank.csv'))

        df = pandas.read_csv(data_path, sep=';')
        print(df.columns)
        #        shape = df.shape

        #        df = df.loc[np.random.choice(df.index, n, replace=False)]
        sex = df['marital'].astype(str).values
        sens_attributes = list(set(sex))

        if 'unknown' in sens_attributes:
            sens_attributes.remove('unknown')

        df1 = df.loc[df['marital'] == sens_attributes[0]]
        df2 = df.loc[df['marital'] == sens_attributes[1]]
        df3 = df.loc[df['marital'] == sens_attributes[2]]

        df = [df1, df2, df3]
        df = pandas.concat(df)

        sex = df['marital'].astype(str).values

        df = df[['age', 'duration', 'euribor3m', 'nr.employed', 'cons.price.idx', 'campaign']].values
        # df = df[['age','duration','balance']].values

        sens_attributes = list(set(sex))
        sex_num = np.zeros(df.shape[0], dtype=int)
        sex_num[sex == sens_attributes[1]] = 1
        sex_num[sex == sens_attributes[2]] = 2

        data = np.array(df, dtype=float)



    elif name == 'CensusII':
        _path = 'USCensus1990raw.data.txt'
        data_path = os.path.join(data_dir, _path)

        if (not os.path.exists(data_path)):

            print('CensusII dataset does not exist in current folder --- Have to download it')
            r = requests.get(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990raw.data.txt',
                allow_redirects=True)
            if r.status_code == requests.codes.ok:
                print('Download successful')
            else:
                print('Could not download - please download')
                sys.exit()

            open(data_path, 'wb').write(r.content)
        df = pandas.read_csv(data_path, sep='\t', header=None)
        # df = pandas.read_csv(data_path,sep=',').iloc[0:,1:]
        sex_num = df.iloc[:, 112].astype(int).values
        selected_attributes = [12, 35, 36, 47, 53, 54, 55, 58, 60, 63, 64, 65, 73, 80, 81, 93, 94, 95, 96, 101, 109,
                               116, 118, 122, 124]
        df = df.iloc[:, selected_attributes].values

        # sens_attributes = list(set(sex_num))
        # df = df.drop(columns='iSex')
        data = np.array(df, dtype=float)
        # data = scale(data, axis = 0)
        K = 20

    ### END CODE OF THE ORIGINAL AUTHORS ###
    ### OUR OWN CODE ###
    elif name == 'Student':
        # Read the Student Dataset (modded version: in sex column: 'M' = 0, 'F' = 1)
        _path = 'student_mat_Cortez_sexmod.csv'
        data_path = os.path.join(data_dir, _path)

        df, n = check_and_read(data_path)

        # Sensitive attribute is sex here, drop this from the dataset
        sex = df['sex']
        sens_attributes = list(set(sex.astype(str).values))  # [0, 1]
        df = df.drop(columns=['sex'])
        sex_num = np.zeros(n, dtype=int)
        sex_num[sex.astype(str).values == sens_attributes[1]] = 1

        # Drop non-numerical features and normalise data
        cont_types = np.where(df.dtypes == 'int64')[0]
        df = df.iloc[:, cont_types]
        data = np.array(df.values, dtype=float)

        # Create 2 clusters
        K = 2

    elif name == 'German_Credit':
        # Read the German Credit Dataset
        _path = 'german_credit.csv'
        data_path = os.path.join(data_dir, _path)

        df, n = check_and_read(data_path)

        # Sensitive attribute is sex here, drop this from the dataset
        sex = df['Sex']
        sens_attributes = list(set(sex.astype(str).values))  # ['male', 'female']
        df = df.drop(columns=['Sex'])
        sex_num = np.zeros(n, dtype=int)
        sex_num[sex.astype(str).values == sens_attributes[1]] = 1

        # Drop non-numerical features and normalise data
        cont_types = np.where(df.dtypes == 'int64')[0]
        df = df.iloc[:, cont_types]
        data = np.array(df.values, dtype=float)

        # Create 2 clusters
        K = 2

    elif name == 'Drugnet':
        # Read the Drugnet Dataset
        _path = 'DRUGATTR.csv'
        data_path = os.path.join(data_dir, _path)

        df, n = check_and_read(data_path)

        # Remove all rows of unknown gender
        remove_ids = [191, 201, 204, 265, 266, 273, 288]
        df.drop(remove_ids, inplace=True)

        # Sensitive attribute is gender here, drop this from the dataset
        sex = df['Gender']
        sens_attributes = list(set(sex.astype(str).values))  # [0, 1, 2]
        df = df.drop(columns=['Gender'])
        sex_num = np.zeros(n, dtype=int)
        sex_num[sex.astype(str).values == sens_attributes[1]] = 1

        # Drop non-numerical features and normalise data
        cont_types = np.where(df.dtypes == 'int64')[0]
        df = df.iloc[:, cont_types]
        data = np.array(df.values, dtype=float)

        # Create 10 clusters
        K = 10

    elif name == 'bank_red' or name == 'bank_2500' or name == 'bank_5000' or name == 'bank_10000' or 'bank_15000':
        # Read the Reduced Bank Dataset, or subsets of the Bank Dataset
        _path = name + '.csv'
        data_path = os.path.join(os.getcwd(), 'data', 'Bank', _path)

        df, _ = check_and_read(data_path, sep=';', header=0)

        # Sensitive attribute is marital here for testing
        marital = df['marital'].astype(str).values
        sens_attributes = list(set(marital))

        # Remove unknown marital statuses
        if 'unknown' in sens_attributes:
            sens_attributes.remove('unknown')

        # Sort the dataset
        df1 = df.loc[df['marital'] == sens_attributes[0]]
        df2 = df.loc[df['marital'] == sens_attributes[1]]
        df3 = df.loc[df['marital'] == sens_attributes[2]]

        df = [df1, df2, df3]
        df = pandas.concat(df)

        marital = df['marital'].astype(str).values

        df = df[['age', 'duration', 'euribor3m', 'nr.employed', 'cons.price.idx', 'campaign']].values

        sens_attributes = list(set(marital))
        sex_num = np.zeros(df.shape[0], dtype=int)
        sex_num[marital == sens_attributes[1]] = 1
        sex_num[marital == sens_attributes[2]] = 2

        data = np.array(df, dtype=float)

        # Create 2 clusters
        K = 2
    ### END OF OUR OWN CODE ###
    ### CODE OF THE ORIGINAL AUTHORS ###
    else:
        pass

    return data, sex_num, K


if __name__ == '__main__':
    dataset = 'CensusII'
    datas = np.load('../data/CensusII.npz')['data']
    X_org = datas['X_org']
    demograph = datas['demograph']

    V_list = [np.array(demograph == j) for j in np.unique(demograph)]
    V_sum = [x.sum() for x in V_list]
    print('Balance of the dataset {}'.format(min(V_sum) / max(V_sum)))
    u_V = [x / X_org.shape[0] for x in V_sum]
