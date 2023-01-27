import numpy as np
import torch
import pandas as pd
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader, Dataset, Subset
# Show 10 decimal places
torch.set_printoptions(precision=8)
np.set_printoptions(precision=8)

'''dataset'''

class kiglis_dataset(Dataset):
    
    def __init__(self, input_data, target_data):
        self.input = input_data
        self.target = target_data
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        inp = self.input[idx]
        tar = self.target[idx]
        return inp, tar


'''kiglis class'''

class KiglisLoader():

    def __init__(self, x_path, y_path, norm):
        self.x_path = x_path
        self.y_path = y_path
        # norm=0: dont normalize
        # norm=1: scale to N(0,1)
        # norm=2: scale to [0,1] (xi-x_min)/(x_max-x_min)
        self.norm = norm

    '''load original data'''
    def load_original_data(self):
        ori_x_data = np.array(pd.read_csv(self.x_path,delimiter=',',header=None))
        ori_y_data = np.array(pd.read_csv(self.y_path,delimiter=',',header=None))
        print("original x_data has shape of: {}".format(ori_x_data.shape))
        print("original y_data has shape of: {}".format(ori_y_data.shape))
        return ori_x_data, ori_y_data

    '''normalize data'''
    def normalize_dataset(self, data, norm=0):
        
        # norm=0: dont normalize
        new_data = torch.from_numpy(data)
        # norm=1: scale to N(0,1)
        if norm == 1:
            mean = torch.mean(new_data, axis=0, keepdims=True)
            std = torch.std(new_data, axis=0, keepdims=True)
            new_data = ((new_data - mean) / std)
            data = new_data
        # norm=2: scale to [0,1] (xi-x_min)/(x_max-x_min)
        if norm == 2:
            new_data = (data-torch.min(data,axis=0))/(torch.max(data,axis=0)-torch.min(data,axis=0))
        return new_data

    '''new data'''
    def new_data(self,norm_x_data, norm_y_data, ori_data_shape, m=20):
        # m: half window size, the whole window has size of (2m+1)
        # for now set m=20
        # ori_len_seq,state_size = ori_x_data.shape = 32768,30
        ori_len_seq = ori_data_shape[0]
        state_size = ori_data_shape[1]
        # x_data: (time_epoch, window_size, state) = (32768-2m, 2m+1, 30)
        # y_data: (time_epoch, state) = (32768-2m, 30)
        x_data = torch.zeros(ori_len_seq-2*m,2*m+1,state_size)
        y_data = torch.zeros(ori_len_seq-2*m,state_size)
        print("x_data has shape: {}".format(x_data.shape))
        print("y_data has shape: {}".format(y_data.shape))

        for i,j in zip(range(x_data.shape[0]),range(m,ori_len_seq-m)):
            x_data[i] = norm_x_data[j-m:j+m+1,:]
            y_data[i] = norm_y_data[j,:]
        
        return x_data, y_data

    '''data loader'''
    def data_loader(self, data_base, Len, BatchSize = 128, split = [0.6,0.2]):

        # Len: the length of input data
        # BatchSize: batch size
        # split: the ratio of train set and validation set, test set is 1-train-val

        train_size = int(Len*split[0])
        val_size = int(Len*split[1])
        train_idx = range(train_size)
        val_idx = range(train_size, train_size+val_size)
        test_idx = range(train_size+val_size, Len)
        # split dataset into train_set, validation_set, test_set
        train_db = Subset(data_base, train_idx)
        val_db = Subset(data_base, val_idx)
        test_db = Subset(data_base, test_idx)

        # create data_loaders
        train_loader = DataLoader(train_db, batch_size=BatchSize, shuffle=False)
        val_loader = DataLoader(val_db, batch_size=BatchSize, shuffle=False)
        test_loader = DataLoader(test_db, batch_size=BatchSize, shuffle=False)

        # show structure of data_loader
        print("length of train_loader: {}".format(len(train_loader)))
        element = next(iter(train_loader))
        print("element in train_loader is: {} with length {}".format(type(element),len(element)))
        input_batch = element[0]
        target_batch = element[1]
        print("one single input batch has shape {}".format(input_batch.shape))
        print("one single target batch has shape {}".format(target_batch.shape))

        return train_loader, val_loader, test_loader

    def create(self):
        # load original data
        ori_x_data, ori_y_data = self.load_original_data()
        # normalize original data
        norm_x_data = self.normalize_dataset(ori_x_data)
        norm_y_data = self.normalize_dataset(ori_y_data)
        # reshape original data
        x_data, y_data = self.new_data(norm_x_data,norm_y_data, ori_x_data.shape)
        # create dataset
        data_base = kiglis_dataset(x_data, y_data)
        # create dataloaders
        train_loader, val_loader, test_loader = self.data_loader(data_base, len(x_data))
        
        return train_loader, val_loader, test_loader, y_data

if __name__ == "__main__":
    norm=2
    x_path = '/home/hardli/python/KIT AIFB HIWI/Interview/kiglis/x_data.txt'
    y_path = '/home/hardli/python/KIT AIFB HIWI/Interview/kiglis/y_data.txt'
    kiglis_loader_creator = KiglisLoader(x_path, y_path, norm)
    train_loader, val_loader, test_loader, y_data = kiglis_loader_creator.create()