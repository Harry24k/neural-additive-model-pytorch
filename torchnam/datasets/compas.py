"""
Modified from
https://github.com/google-research/google-research/blob/master/neural_additive_models/data_utils.py
License
http://www.apache.org/licenses/LICENSE-2.0
"""
import zipfile
import os
from urllib.request import urlretrieve
from shutil import copyfile

import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive
import torch
from torch.utils.data import TensorDataset

class COMPAS() :
    def __init__(self, root="./data_nam", transform=None, train_test_rate = 0.8) :
        self.train_test_rate = train_test_rate
        if root[-1] == "/" :
            root = root[:-1]
        
        self._ensure_dataset_loaded(root)

    def _ensure_dataset_loaded(self, root):

#         score_csv = 'compas-scores-raw.csv'
        violent_csv = 'cox-violent-parsed' ## data 전처리 되어있는 형태, 

        if root[-1] == "/":
            root = root[:-1]

        if os.path.isfile(root+"/"+violent_csv):#os.path.isfile(root+"/"+score_csv) and 
            pass
        else:
#             raise ValueError("Please download Credit Fraud csv file via https://www.kaggle.com/datasets/danofer/compass?select=cox-violent-parsed.csv")
            raise ValueError("Please download Credit Fraud csv file via https://www.openml.org/search?type=data&sort=runs&id=44162&status=active")
        file = open(root+"/"+violent_csv)
        Lines = file.readlines()
        data_list = []
        for line in Lines:
            if not("%" in line or "@" in line or line == '\n'):
                data_list.append(list(map(float, line.strip().split(","))))
#         score_data = pd.read_csv(score_csv)
#         violent_data = pd.read_csv(violent_csv)

        total_tensor = torch.tensor(data_list)
        
        # total_tensor = total_tensor[total_tensor[:, 6] < 5]  # remove juv_other_count >= 5
            
            
        compas_data = total_tensor[:, :-1].to(torch.float)
        compas_target = (total_tensor[:, -1]+1).to(torch.long)
            

        ds = TensorDataset(compas_data, compas_target)
        self.train_data, self.test_data = torch.utils.data.random_split(ds, [int(len(ds)*self.train_test_rate), len(ds) - int(len(ds)*self.train_test_rate)])
            
    