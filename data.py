import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sampler import BalancedBatchSampler
import pandas as pd
import numpy as np
import random

pd.set_option('mode.chained_assignment', None)


class data_loader():
    def __init__(self, args, s):

        # bank data --> sep=';'
        # adult, home data --> sep=','
        # german data --> sep=' '
        if args.dataset == 'german':
            train_path = 'german-data/german.train'

        elif args.dataset == 'german-pre-dp':
            if s == 0:
                train_path = 'german-data/german.train'
            else:
                train_path = 'german-data/synth/syth_data_correlated_{}.csv'.format(s)

        if args.dataset == 'bank':
            train_path = 'bank-data/bank-additional-full.csv'

        elif args.dataset == 'bank-pre-dp':
            if s == 0:
                train_path = 'bank-data/bank-additional-full.csv'
            else:
                train_path = 'bank-data/synth/syth_data_correlated_ymod_{}.csv'.format(s)

        elif args.dataset == 'adult':
            train_path = 'adult-data/adult.data'

        elif args.dataset == 'adult-pre-dp':
            if s == 0:
                train_path = 'adult-data/adult.data'
            else:
                train_path = 'adult-data/synth/syth_data_correlated_ymod_{}.csv'.format(s)

        elif args.dataset == 'home':
            train_path = 'home-data/hcdf_train.csv'

        elif args.dataset == 'home-pre-dp':
            if s == 0:
                train_path = 'home-data/hcdf_train.csv'
            else:
                train_path = 'home-data/synth/syth_data_correlated_ymod_{}.csv'.format(s)

        if args.dataset == 'german' or args.dataset == 'german-pre-dp':
            cols = ['existing_checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
                    'savings', 'employment_since', 'installment_rate', 'status_sex', 'other_debtors',
                    'residence_since', 'property', 'age', 'other_installment_plans', 'housing',
                    'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'y']

            test_path = 'german-data/german.test'
            if args.dataset == 'german-pre-dp'and s > 0:
                sep = ','
            else:
                sep = ' '
            train_df = pd.read_csv(train_path, sep=sep, names=cols)
            test_df = pd.read_csv(test_path, sep=' ', names=cols)

            train_df['y'] = train_df['y'].apply(lambda x: 0 if x == 2 else 1)
            test_df['y'] = test_df['y'].apply(lambda x: 0 if x == 2 else 1)
            print(train_df)

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

        elif args.dataset == 'home' or args.dataset == 'home-pre-dp':
            test_path = 'home-data/hcdf_test.csv'

            train_df = pd.read_csv(train_path, sep=',', header=0)
            test_df = pd.read_csv(test_path, sep=',', header=0)

            train_df = train_df.drop(columns=['FLAG_OWN_CAR'])
            test_df = test_df.drop(columns=['FLAG_OWN_CAR'])

            train_df = train_df.rename(columns={"TARGET": "y", "CODE_GENDER": "GENDER"})
            test_df = test_df.rename(columns={"TARGET": "y", "CODE_GENDER": "GENDER"})

        train_df = train_df.dropna()
        test_df = test_df.dropna()

        train_df = train_df.sample(frac=1).reset_index(drop=True)  # shuffle df
        #test_df = test_df.sample(frac=1).reset_index(drop=True)  # shuffle df

        if args.num_teachers == 0 or s == 0:
            train_data = LoadDataset(train_df, args.dataset, args.sensitive)
            test_data = LoadDataset(test_df, args.dataset, args.sensitive)

            self.sensitive_keys = train_data.getkeys()
            self.train_size = len(train_data)
            self.test_size = len(test_data)
            self.sensitive_col_idx = train_data.get_sensitive_idx()
            self.cat_emb_size = train_data.categorical_embedding_sizes  # size of categorical embedding
            print("***", self.cat_emb_size)
            self.num_conts = train_data.num_numerical_cols  # number of numerical variables

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
        else:
            student_train_size = int(len(train_df) * .3)
            teacher_train_df = train_df.iloc[student_train_size:, :]
            student_train_df = train_df.iloc[:student_train_size, :]


            train_data = LoadDataset(teacher_train_df, args.dataset, args.sensitive)
            student_train_data = LoadDataset(student_train_df, args.dataset, args.sensitive)
            test_data = LoadDataset(test_df, args.dataset, args.sensitive)

            self.sensitive_keys = train_data.getkeys()
            self.train_size = len(train_data)
            self.test_size = len(test_data)
            self.sensitive_col_idx = train_data.get_sensitive_idx()

            student_train_size = len(student_train_data)

            self.cat_emb_size = train_data.categorical_embedding_sizes  # size of categorical embedding
            #print(self.cat_emb_size)
            self.num_conts = train_data.num_numerical_cols  # number of numerical variables

            class_count = dict(train_df.y.value_counts())
            class_weights = [value / len(train_data) for _, value in class_count.items()]

            train_batch = args.batch_size
            test_batch = len(test_data)

            self.teacher_loaders = []
            data_size = self.train_size // args.num_teachers


            for i in range(args.num_teachers):
                indices = list(range(i * data_size, (i + 1) * data_size))

                subset_data = Subset(train_data, indices)
                subset_data_Y = [i[2] for i in subset_data]

                subset_data_Y = torch.stack(subset_data_Y)

                loader = DataLoader(dataset=subset_data,
                                    sampler=BalancedBatchSampler(subset_data, subset_data_Y),
                                    batch_size=train_batch)

                self.teacher_loaders.append(loader)

            """
            indices = list(range(len(test_data)))
            indices = random.sample(indices, len(indices))
            student_split = int(len(test_data) * .7)
            
            student_train_data = Subset(test_data, indices[:student_split])
            student_test_data = Subset(test_data, indices[student_split+1:])
            """
            self.student_train_loader = torch.utils.data.DataLoader(student_train_data,
                                                                    #sampler=BalancedBatchSampler(student_train_data,
                                                                    #                             student_train_data.Y),
                                                                    batch_size=student_train_size)
            self.student_test_loader = torch.utils.data.DataLoader(test_data,
                                                                   batch_size=test_batch)

    def getkeys(self):
        return self.sensitive_keys

    def get_input_properties(self):
        return self.cat_emb_size, self.num_conts

    def __getitem__(self):
        return self.train_loader, self.test_loader

    def __len__(self):
        return self.train_size, self.test_size

    def train_teachers(self):
        return self.teacher_loaders

    def student_data(self):
        return self.student_train_loader, self.student_test_loader

    def get_sensitive_idx(self):
        return  self.sensitive_col_idx

class LoadDataset(Dataset):
    def __init__(self, data, mode, sensitive_col):

        self.len = data.shape[0]

        print(data.head())

        if mode == 'german' or mode == 'german-pre-dp':

            categorical_columns = ['existing_checking',
                                   'credit_history',
                                   'purpose',
                                   'savings',
                                   'employment_since',
                                   'status_sex',
                                   'other_debtors',
                                   'property',
                                   'other_installment_plans',
                                   'housing',
                                   'job',
                                   'telephone',
                                   'foreign_worker']

            numerical_columns = ['duration', 'credit_amount',
                                 'installment_rate',
                                 'residence_since', 'age',
                                 'existing_credits', 'people_liable']

            # categorical variables
            for category in categorical_columns:
                data[category] = data[category].astype('category')

            existing_checking = data['existing_checking'].cat.codes.values
            credit_history = data['credit_history'].cat.codes.values
            purpose = data['purpose'].cat.codes.values
            savings = data['savings'].cat.codes.values
            employment_since = data['employment_since'].cat.codes.values
            status_sex = data['status_sex'].cat.codes.values
            other_debtors = data['other_debtors'].cat.codes.values
            property = data['property'].cat.codes.values
            other_installment_plans = data['other_installment_plans'].cat.codes.values
            housing = data['housing'].cat.codes.values
            job = data['job'].cat.codes.values
            telephone = data['telephone'].cat.codes.values
            foreign_worker = data['foreign_worker'].cat.codes.values

            #self.cat_dict = dict(enumerate(data['job'].cat.categories))  # 10
            self.cat_dict = dict(enumerate(data[sensitive_col].cat.categories))  # 5
            self.sensitive_col_idx = categorical_columns.index(sensitive_col)
            print(self.cat_dict)

            categorical_data = np.stack([existing_checking, credit_history, purpose,
                                         savings, employment_since, status_sex,
                                         other_debtors, property,
                                         other_installment_plans, housing,
                                         job, telephone, foreign_worker], 1)

        if mode == 'bank' or mode == 'bank-pre-dp':
            categorical_columns = ['job', 'marital',
                                   'education', 'default', 'housing',
                                   'loan', 'contact', 'month',
                                   'day_of_week', 'poutcome']

            numerical_columns = ['duration', 'campaign', 'pdays',
                                 'previous', 'emp.var.rate', 'cons.price.idx',
                                 'cons.conf.idx', 'euribor3m', 'nr.employed']

            # categorical variables
            for category in categorical_columns:
                data[category] = data[category].astype('category')

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
            #            self.cat_dict = dict(enumerate(data['education'].cat.categories)) # 2
            self.cat_dict = dict(enumerate(data[sensitive_col].cat.categories))  # 0
            self.sensitive_col_idx = categorical_columns.index(sensitive_col)

            #            self.cat_dict = dict(enumerate(data['marital'].cat.categories)) # 1
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

            numerical_columns = ['education-num', 'capital-gain',
                                 'capital-loss', 'hours-per-week']

            # categorical variables
            for category in categorical_columns:
                data[category] = data[category].astype('category')

            workclass = data['workclass'].cat.codes.values
            education = data['education'].cat.codes.values
            marital_status = data['marital-status'].cat.codes.values
            occupation = data['occupation'].cat.codes.values
            relationship = data['relationship'].cat.codes.values
            race = data['race'].cat.codes.values
            sex = data['sex'].cat.codes.values
            native_country = data['native-country'].cat.codes.values
            self.cat_dict = dict(enumerate(data[sensitive_col].cat.categories)) # 1
            #self.cat_dict = dict(enumerate(data['race'].cat.categories))  # 5
            #            self.cat_dict = dict(enumerate(data['marital-status'].cat.categories)) # 2
            self.sensitive_col_idx = categorical_columns.index(sensitive_col)

            print(self.cat_dict)

            categorical_data = np.stack([workclass, education, marital_status,
                                         occupation, relationship, race,
                                         sex, native_country], 1)

        elif mode == 'home' or mode == 'home-pre-dp':

            # define data column types
            categorical_columns = ["NAME_CONTRACT_TYPE", "GENDER",
                                    "FLAG_OWN_REALTY",
                                   "NAME_TYPE_SUITE", "NAME_INCOME_TYPE",
                                   "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
                                   "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
                                   "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE",
                                   "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
                                   "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]

            numerical_columns = set(data.columns)
            numerical_columns.remove('y')
            numerical_columns.difference_update(categorical_columns)

            print(data.head())
            # categorical variables
            for category in categorical_columns:
                data[category] = data[category].astype('category')

            contract = data['NAME_CONTRACT_TYPE'].cat.codes.values
            gender = data['GENDER'].cat.codes.values
            realty = data['FLAG_OWN_REALTY'].cat.codes.values
            suite = data['NAME_TYPE_SUITE'].cat.codes.values
            income = data['NAME_INCOME_TYPE'].cat.codes.values
            education = data['NAME_EDUCATION_TYPE'].cat.codes.values
            family = data['NAME_FAMILY_STATUS'].cat.codes.values
            housing = data['NAME_HOUSING_TYPE'].cat.codes.values
            occupation = data['OCCUPATION_TYPE'].cat.codes.values
            process = data['WEEKDAY_APPR_PROCESS_START'].cat.codes.values
            organization = data['ORGANIZATION_TYPE'].cat.codes.values
            fondkaprement = data['FONDKAPREMONT_MODE'].cat.codes.values
            housetype = data['HOUSETYPE_MODE'].cat.codes.values
            wallsmaterial = data['WALLSMATERIAL_MODE'].cat.codes.values
            emergencystate = data['EMERGENCYSTATE_MODE'].cat.codes.values

            self.cat_dict = dict(enumerate(data[sensitive_col].cat.categories)) # 1
            #self.cat_dict = dict(enumerate(data['race'].cat.categories))  # 5
            #            self.cat_dict = dict(enumerate(data['marital-status'].cat.categories)) # 2
            self.sensitive_col_idx = categorical_columns.index(sensitive_col)

            #print(self.cat_dict)

            categorical_data = np.stack([contract,
                                         gender,
                                         realty,
                                         suite,
                                         income,
                                         education,
                                         family,
                                         housing,
                                         occupation,
                                         process,
                                         organization,
                                         fondkaprement,
                                         housetype,
                                         wallsmaterial,
                                         emergencystate
                                        ], 1)


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

    def get_sensitive_idx(self):
        return  self.sensitive_col_idx

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
