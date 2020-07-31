from lib.dataset.Base_DataSet import Base_DataSet
import os
import h5py
import numpy as np
from skimage import transform


class AlignDataSet(Base_DataSet):
    def __init__(self, dataset_dir):
        super(AlignDataSet, self).__init__()
        self.ext = '.h5'
        self.dataset_root = dataset_dir
        self.data_list = os.listdir(self.dataset_root)
        self.dataset_size = len(self.data_list)
    
    @property
    def name(self):
        return 'AlignDataSet'

    @property
    def num_samples(self):
        return self.dataset_size
    
    def get_data_path(self, root, index_name):
        data_path = os.path.join(root, index_name)
        assert os.path.exists(data_path), 'Path do not exist: {}'.format(data_path)
        return data_path
    
    def load_file(self, data_path):
        hdf5 = h5py.File(data_path, 'r')
        input_drr_ap = np.asarray(hdf5['input_drr_ap'])
        input_xray_ap = np.asarray(hdf5['input_xray_ap'])
        input_drr_lat = np.asarray(hdf5['input_drr_lat'])
        input_xray_lat = np.asarray(hdf5['input_xray_lat'])
        correspondence_2D_ap = np.asarray(hdf5['correspondence_2D_ap'])
        correspondence_2D_lat = np.asarray(hdf5['correspondence_2D_lat'])
        fiducial_3D = np.asarray(hdf5['fiducial_3D'])
        # input_drr1 = transform.resize(input_drr1, (64, 64))
        # input_drr2 = transform.resize(input_drr2, (64, 64))
        # correspondence_2D = correspondence_2D / (200 / 64)
        # correspondence_2D = correspondence_2D.astype(np.int64)
        input_drr_ap = np.expand_dims(input_drr_ap, 0)
        input_xray_ap = np.expand_dims(input_xray_ap, 0)
        input_drr_lat = np.expand_dims(input_drr_lat, 0)
        input_xray_lat = np.expand_dims(input_xray_lat, 0)
        correspondence_2D_ap = np.expand_dims(correspondence_2D_ap, 0)
        correspondence_2D_lat = np.expand_dims(correspondence_2D_lat, 0)
        hdf5.close()
        return input_drr_ap, input_xray_ap, correspondence_2D_ap, input_drr_lat, input_xray_lat, correspondence_2D_lat, fiducial_3D 

    def preprocess(self, input_drr):
        # Normalization
        input_drr = (input_drr - np.min(input_drr)) / (np.max(input_drr) - np.min(input_drr))
        return input_drr

    '''
    generate batch
    '''
    def pull_item(self, item):
        data_path = self.get_data_path(self.dataset_root, self.data_list[item])
        input_drr_ap, input_xray_ap, correspondence_2D_ap, input_drr_lat, input_xray_lat, correspondence_2D_lat, fiducial_3D = \
            self.load_file(data_path)
        input_drr_ap = self.preprocess(input_drr_ap)
        input_xray_ap = self.preprocess(input_xray_ap)
        input_drr_lat = self.preprocess(input_drr_lat)
        input_xray_lat = self.preprocess(input_xray_lat)

        return input_drr_ap, input_xray_ap, correspondence_2D_ap, input_drr_lat, input_xray_lat, correspondence_2D_lat, fiducial_3D

    

