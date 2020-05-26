from torch.utils.data import Dataset


class Base_DataSet(Dataset):
    '''
    Base DataSet
    '''
    def __init__(self):
        pass

    def __getitem__(self, item):
        return self.pull_item(item)
    
    def __len__(self):
        return self.num_samples

    @property
    def name(self):
        return 'Base DataSet'

    @property
    def num_samples(self):
        return 1
    
    def pull_item(self, *str):
        pass