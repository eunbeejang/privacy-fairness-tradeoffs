from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sampler import BalancedBatchSampler
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
pd.set_option('mode.chained_assignment', None)
import fairlearn.metrics as flm
import sklearn.metrics as skm


class data_loader():
    def __init__(self, args):
        train_path = args.train_data_path
        train_df = pd.read_csv(train_path, sep=';')
        train_df = train_df.sample(frac=1).reset_index(drop=True)   # shuffle df

        test_path = args.test_data_path
        test_df = pd.read_csv(test_path, sep=';')
        test_df = test_df.sample(frac=1).reset_index(drop=True) # shuffle df
        #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

        train_data = BankDataset(train_df)
        test_data = BankDataset(test_df)

        self.train_size = len(train_data)
        self.test_size = len(test_data)

        self.cat_emb_size = train_data.categorical_embedding_sizes # size of categorical embedding
        self.num_conts = train_data.num_numerical_cols # number of numerical variables

        """
        class_count = [i for i in get_class_distribution(train_data.y).values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        """
        class_count = dict(train_df.y.value_counts())
        class_weights = [value / len(train_data) for _, value in class_count.items()]

        self.train_loader = DataLoader(dataset=train_data,
                                       sampler=BalancedBatchSampler(train_data, train_data.Y),
                                       batch_size=args.batch_size)
                                  #sampler=WeightedRandomSampler(class_weights, args.batch_size, replacement=True),
                                  #drop_last=True)

        self.test_loader = DataLoader(dataset=test_data,
                                 batch_size=args.test_batch_size,
                                 drop_last=True)
    def __getitem__(self):
        return self.train_loader, self.test_loader, self.cat_emb_size, self.num_conts

    def __len__(self):
        return self.train_size, self.test_size


class BankDataset(Dataset):
    def __init__(self, data):

        print("\nDataset shape: ", data.shape)
        print("Check for NaNs: {}\n".format(data.isnull().values.any()))
        self.len = data.shape[0]

        # define data column types
        categorical_columns = ['age', 'job', 'marital',
                               'education', 'default', 'housing',
                               'loan', 'contact', 'month',
                               'day_of_week', 'poutcome']

        numerical_columns = ['duration', 'campaign', 'pdays',
                             'previous', 'emp.var.rate', 'cons.price.idx',
                             'cons.conf.idx', 'euribor3m', 'nr.employed']

        print(data.head)
        print(data.dtypes)

        # categorical variables
        for category in categorical_columns:
            data[category] = data[category].astype('category')
        print(data.dtypes)

        age = data['age'].cat.codes.values
        job = data['job'].cat.codes.values
        marital = data['marital'].cat.codes.values
        education = data['education'].cat.codes.values
        default = data['default'].cat.codes.values
        housing = data['housing'].cat.codes.values
        loan = data['loan'].cat.codes.values
        contact = data['contact'].cat.codes.values
        month = data['month'].cat.codes.values
        day_of_week = data['day_of_week'].cat.codes.values
        poutcome = data['poutcome'].cat.codes.values

        categorical_data = np.stack([age, job, marital,
                                     education, default, housing,
                                     loan, contact, month,
                                     day_of_week, poutcome], 1)
        self.categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

        # continuous variables
        numerical_data = np.stack([data[col].values for col in numerical_columns], 1)
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float64)

        # target variable
        data['y'] = data['y'].astype('category')
        Y = data['y'].cat.codes.values
        self.Y = torch.tensor(Y.flatten(), dtype=torch.int64)

        # define categorical and continuous embedding sizes
        categorical_column_sizes = [len(data[column].cat.categories) for column in categorical_columns]
        self.categorical_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2))
                                            for col_size in categorical_column_sizes]
        self.num_numerical_cols = self.numerical_data.shape[1]

    def __getitem__(self, idx):
        return self.categorical_data[idx], self.numerical_data[idx].float(), self.Y[idx].float()

    def __len__(self):
        return self.len


def get_class_distribution(dataset_obj):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}

    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1

    return count_dict