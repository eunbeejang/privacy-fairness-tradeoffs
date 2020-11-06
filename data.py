from torch.utils.data import Dataset, DataLoader
#from torch.utils.data.sampler import WeightedRandomSampler
from sampler import BalancedBatchSampler
import torch
import pandas as pd
import numpy as np
import os
pd.set_option('mode.chained_assignment', None)


class data_loader():
    def __init__(self, args, s):

        # bank data --> sep=';'
        # adult data --> sep=','
        if args.dataset == 'bank':
            train_path = 'bank-data/bank-additional-full.csv'

        elif args.dataset == 'bank-pre-dp':
            if s == 0.1:
                train_path = 'bank-data/out/sythetic_data01.csv'
            elif s == 0.5:
                train_path = 'bank-data/out/sythetic_data05.csv'
            elif s == 1.0:
                train_path = 'bank-data/out/sythetic_data.csv'
            else:
                train_path = 'bank-data/bank-additional-full.csv'

        elif args.dataset == 'adult':
            train_path = 'adult-data/adult.data'

        elif args.dataset == 'adult-pre-dp':
            if s == 0.1:
                train_path = 'adult-data/out/sythetic_data01.csv'
            elif s == 0.5:
                train_path = 'adult-data/out/sythetic_data05.csv'
            elif s == 1.0:
                train_path = 'adult-data/out/sythetic_data.csv'
            else:
                train_path = 'adult-data/adult.data'


        if args.dataset == 'bank' or args.dataset == 'bank-pre-dp':
            cols = ['age', 'job', 'marital', 'education',
                    'default', 'housing', 'loan', 'contact',
                    'month', 'day_of_week', 'duration',
                    'campaign', 'pdays', 'previous', 'poutcome',
                    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
                    'nr.employed', 'y']
            test_path = 'bank-data/bank-additional.csv'
            train_df = pd.read_csv(train_path, sep=';', names=cols)
            test_df = pd.read_csv(test_path, sep=';', names=cols)

        elif args.dataset == 'adult' or args.dataset == 'adult-pre-dp':
            test_path = 'adult-data/adult.test'

            cols = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country', 'y']

            train_df = pd.read_csv(train_path, sep=',', names=cols)
            test_df = pd.read_csv(test_path, sep=',', names=cols)

            train_df = train_df.replace({'?': np.nan})
            test_df = test_df.replace({'?': np.nan})

            train_df['y'] = train_df['y'].apply(lambda x: 0 if ">50K" in x else 1)
            test_df['y'] = test_df['y'].apply(lambda x: 0 if ">50K" in x else 1)



        train_df = train_df.dropna()
        test_df = test_df.dropna()

        train_df = train_df.sample(frac=1).reset_index(drop=True)  # shuffle df
        test_df = test_df.sample(frac=1).reset_index(drop=True)  # shuffle df

        train_data = LoadDataset(train_df, args.dataset)
        test_data = LoadDataset(test_df, args.dataset)

        self.sensitive_keys = train_data.getkeys()
        self.train_size = len(train_data)
        self.test_size = len(test_data)

        self.cat_emb_size = train_data.categorical_embedding_sizes # size of categorical embedding
        print(self.cat_emb_size)
        self.num_conts = train_data.num_numerical_cols # number of numerical variables



        class_count = dict(train_df.y.value_counts())
        class_weights = [value / len(train_data) for _, value in class_count.items()]

        train_batch = args.batch_size
        test_batch = len(test_data)

        self.train_loader = DataLoader(dataset=train_data,
                                       sampler=BalancedBatchSampler(train_data, train_data.Y),
                                       batch_size=train_batch)

        self.test_loader = DataLoader(dataset=test_data,
                                      batch_size=test_batch,
                                 drop_last=True)



    def train_dataloader(self): # for PyTorch Lightening
        return DataLoader(self.train_loader)


    def test_dataloader(self): # for PyTorch Lightening
        return DataLoader(self, test_loader)

    def getkeys(self):
        return self.sensitive_keys

    def __getitem__(self):
        return self.train_loader, self.test_loader, self.cat_emb_size, self.num_conts

    def __len__(self):
        return self.train_size, self.test_size



class LoadDataset(Dataset):
    def __init__(self, data, mode):


        self.len = data.shape[0]

        print(data.head())

        if mode == 'bank' or mode == 'bank-pre-dp':
            categorical_columns = ['job', 'marital',
                                   'education', 'default', 'housing',
                                   'loan', 'contact', 'month',
                                   'day_of_week', 'poutcome']

            numerical_columns = ['age', 'duration', 'campaign', 'pdays',
                                 'previous', 'emp.var.rate', 'cons.price.idx',
                                 'cons.conf.idx', 'euribor3m', 'nr.employed']

            # categorical variables
            for category in categorical_columns:
                data[category] = data[category].astype('category')

            # age = data['age'].cat.codes.values
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

            self.cat_dict = dict(enumerate(data['education'].cat.categories))
            print(self.cat_dict)

            categorical_data = np.stack([job, marital, education,
                                         default, housing, loan,
                                         contact, month, day_of_week,
                                         poutcome], 1)

        elif mode == 'adult' or mode == 'adult-pre-dp':

            # define data column types
            categorical_columns = ['workclass', 'education', 'marital-status',
                                   'occupation', 'relationship', 'race',
                                   'sex', 'native-country']

            numerical_columns = ['age', 'education-num', 'capital-gain',
                                 'capital-loss', 'hours-per-week']


            # categorical variables
            for category in categorical_columns:
                data[category] = data[category].astype('category')

            workclass = data['workclass'].cat.codes.values
            education = data ['education'].cat.codes.values
            marital_status = data['marital-status'].cat.codes.values
            occupation = data['occupation'].cat.codes.values
            relationship = data['relationship'].cat.codes.values
            race = data['race'].cat.codes.values
            sex = data['sex'].cat.codes.values
            native_country = data['native-country'].cat.codes.values

            self.cat_dict = dict(enumerate(data['education'].cat.categories))
            print(self.cat_dict)

            categorical_data = np.stack([workclass, education, marital_status,
                                         occupation, relationship, race,
                                         sex, native_country], 1)



        self.categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

        # continuous variables
#        numerical_data = np.stack([data[col].values for col in numerical_columns], 1)
#        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float64)
        for numerical in numerical_columns:
            data[numerical] = pd.to_numeric(data[numerical], errors='coerce')
        numerical_data = np.stack([data[col].values.astype(np.float) for col in numerical_columns], 1)
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

    def getkeys(self):
        return self.cat_dict

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



