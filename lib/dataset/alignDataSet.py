from lib.dataset.Base_DataSet import Base_DataSet
import os
import h5py
import numpy as np


class AlignDataSet(Base_DataSet):
    def __init__(self, data_dir):
        super(AlignDataSet, self).__init__()
        self.ext = '.h5'
        self.dataset_paths = data_dir
        self.data_list = os.listdir(self.dataset_paths)
        self.dataset_size = len(self.data_list)
    
    @property
    def name(self):
        return 'AlignDataSet'
    
    @property
    def get_data_path(self):
        path = os.path.join(self.dataset_paths)
    
    @property
    def get_data_path(self, root, index_name):
        data_path = os.path.join(root, index_name)
        assert os.path.exists(data_path), 'Path do not exist: {}'.format(data_path)
        return data_path
    

