import os
import numpy as np
import pandas as pd
from sklearn import datasets as skdsets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset

from .compas import COMPAS
from .td import Datasets


class TabularDatasets(Datasets):
    def _set_data_Housing(self, root, train_transform, test_transform):
        train_test_rate = 0.8
        housing_data = torch.Tensor(skdsets.fetch_california_housing()[
                                    'data']).to(torch.float)
        housing_target = torch.Tensor(
            skdsets.fetch_california_housing()['target'])
        housing_target = torch.unsqueeze(housing_target, dim=1)

        ds = TensorDataset(housing_data, housing_target)
        self.train_data, self.test_data = torch.utils.data.random_split(ds, [int(len(
            ds)*train_test_rate), len(ds) - int(len(ds)*train_test_rate)], generator=torch.Generator().manual_seed(42))
        self.n_features = 8
        self.cols = ["MedInc",
                    "HouseAge" ,
                    "AveRooms" ,
                    "AveBedrms",
                    "Population",
                    "AveOccup" ,
                    "Latitude" ,
                    "Longitude"]

    def _set_data_Boston(self, root, train_transform, test_transform):
        train_test_rate = 0.8
        boston_data = torch.Tensor(skdsets.load_boston()[
                                   'data']).to(torch.float)
        boston_target = torch.Tensor(skdsets.load_boston()['target'])
        boston_target = torch.unsqueeze(boston_target, dim=1)

        ds = TensorDataset(boston_data, boston_target)
        self.train_data, self.test_data = torch.utils.data.random_split(ds, [int(len(
            ds)*train_test_rate), len(ds) - int(len(ds)*train_test_rate)], generator=torch.Generator().manual_seed(42))
        self.n_features = 13
        self.cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT']

    def _set_data_COMPAS(self, root, train_transform, test_transform):
        self.train_data = COMPAS(
            root=root, transform=train_transform).train_data
        self.test_data = COMPAS(root=root, transform=train_transform).test_data
        self.n_features = 17
        self.cols = ['sex',
                     'age',
                     'age_cat',
                     'race',
                     'juv_fel_count',
                     'juv_misd_count',
                     'juv_other_count',
                     'priors_count',
                     'days_b_screening_arrest',
                     'c_days_from_compas',
                     'c_charge_degree',
                     'decile_score',
                     'score_text',
                     'v_type_of_assessment',
                     'v_decile_score',
                     'v_score_text',
                     'end',
                    #  'is_recid', #target
                           ]

    def _set_data_Credit(self, root, train_transform, test_transform):
        csv_data = 'creditcard.csv'

        if root[-1] == "/":
            root = root[:-1]

        if os.path.isfile(root+"/"+csv_data):
            pass
        else:
            raise ValueError(
                "Please download Credit Fraud csv file via https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download")

        train_test_rate = 0.8

        total_data = pd.read_csv(root+"/"+csv_data)
        total_tensor = torch.tensor(total_data.values)

        credit_data = total_tensor[:, :-1].to(torch.float)
        credit_target = total_tensor[:, -1].to(torch.long)

        data_x = credit_data
        data_y = credit_target
        # Stratified Split
        train_idx, test_idx = train_test_split(np.array(range(data_x.shape[0])),
                                               shuffle=True,
                                               stratify=data_y,
                                               test_size=1-train_test_rate,
                                               random_state=42)
        self.train_data = TensorDataset(data_x[train_idx], data_y[train_idx])
        self.test_data = TensorDataset(data_x[test_idx], data_y[test_idx])
        self.n_features = 30
        self.cols = None

    def _set_data_Credit_Seed(self, root, train_transform, test_transform):
        csv_data = 'credit.pt'

        train_data, test_data = torch.load(root+"/"+csv_data)

        x_train, y_train = train_data
        x_test, y_test = test_data

        self.train_data = TensorDataset(
            torch.tensor(x_train), torch.tensor(y_train))
        self.test_data = TensorDataset(
            torch.tensor(x_test), torch.tensor(y_test))
        self.n_features = 30
        self.cols = None

    def _set_data_FICO(self, root, train_transform, test_transform):
        csv_data = 'heloc_dataset_v1.csv'

        if root[-1] == "/":
            root = root[:-1]

        if os.path.isfile(root+"/"+csv_data):
            pass
        else:
            raise ValueError("Please download Credit Fraud csv file ")

        train_test_rate = 0.8

        total_data = pd.read_csv(root+"/"+csv_data)
        total_data = total_data.replace("Good", 1)
        total_data = total_data.replace("Bad", 0)

        total_tensor = torch.tensor(total_data.values)

        fico_data = total_tensor[:, 1:].to(torch.float)
        fico_target = total_tensor[:, 0].to(torch.float)

        data_x = fico_data
        data_y = fico_target
        # Stratified Split
        train_idx, test_idx = train_test_split(np.array(range(data_x.shape[0])),
                                               shuffle=True,
                                               stratify=data_y,
                                               test_size=1-train_test_rate,
                                               random_state=42)
        self.train_data = TensorDataset(data_x[train_idx], data_y[train_idx])
        self.test_data = TensorDataset(data_x[test_idx], data_y[test_idx])
        self.n_features = 23
        self.cols = ['External Risk Estimate', 
                    'Months Since Oldest Trade Open',
                   'Months Since Most Recent Trade', 
                    'Average Months in File', 
                    '# Satisfactory Trades',
                   '# Trades 60+ Ever',
                    '# Trades 90+ Ever',
                   '% Trades Never Delinquent',
                    'Months Since Most \n Recent Delinquency',
                   'Max Delq/Public Records \n Last Year', 
                    'Max Delinquency Ever',
                    '# Total Trades ',
                   '# Trades Open in Last 12 Months',
                    '% Installment Trades',
                   'Months Since Most Recent \n Inquiry excluding 7 days',
                    '# Inquiries in Last 6 Months',
                    '# Inquiries in Last 6 Months \n excluding 7 days',
                   'Net Fraction Revolving Burden',
                    'Net Fraction Installment Burden',
                   '# Revolving Trades with Balance',
                    '# Install Trades with Balance',
                   '# xBank/Natl Trades with \n high utilization ratio', 
                    '% Trades with Balance']