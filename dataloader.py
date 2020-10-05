from torch.utils.data import Dataset, DataLoader
import torch
from torch import from_numpy, tensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BankDataset(Dataset):
    def __init__(self):
        data = pd.read_csv('./bank-data/bank-additional-full.csv', sep=';')
        print("Dataset shape: ", data.shape)
        print("Check for NaNs: ", data.isnull().values.any())


        """
        Visualize data
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 10
        fig_size[1] = 8
        plt.rcParams["figure.figsize"] = fig_size
        
        data.y.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=['skyblue', 'orange'], explode=(0.05, 0.05))
        plt.show()
        sns.countplot(x='y', hue='education', data=data)
        plt.show()
        """


        # DEFINE DATA COLUMN TYPES
        categorical_columns = ['age', 'job', 'marital',
                               'education', 'default', 'housing',
                               'loan', 'contact', 'month',
                               'day_of_week', 'poutcome']

        numerical_columns = ['duration', 'campaign', 'pdays',
                             'previous', 'emp.var.rate', 'cons.price.idx',
                             'cons.conf.idx', 'euribor3m', 'nr.employed']

        output = ['y']


        # CATEGORICAL TENSORS
        for category in categorical_columns:
            data[category] = data[category].astype('category')

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

        print(categorical_data[:10])

        # NUMERICAL TENSORS
        numerical_data = np.stack([data[col].values for col in numerical_columns], 1)
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float64)

        #print(numerical_data[:10])


        # OUTPUT TENSOR
        data['y'] = data['y'].astype('category')
        self.outputs = torch.tensor(data['y'].cat.codes.values.flatten(), dtype=torch.int64)
        #print(outputs[:5])

        # PLOT DATA SHAPE
        print(categorical_data.shape)
        print(numerical_data.shape)
        print(outputs.shape)
        exit()



    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = BankDataset()
train_loader = DataLoader(  dataset=dataset,
                            train=True,
                            batch_size=args.batch_size,
                            shuffle=True,
                            transform=transforms.Compose(
                                [
                                    transforms.ToTensor(),
                                    transforms.Normalize((MEAN,), (STD,)),
                                ]
                            ),
                            **kwargs
                        )

test_loader = DataLoader(   dataset=dataset,
                            train=False,
                            batch_size=args.batch_size,
                            shuffle=True,
                            transform=transforms.Compose(
                                [
                                    transforms.ToTensor(),
                                    transforms.Normalize((MEAN,), (STD,)),
                                ]
                            ),
                            **kwargs
                        )    