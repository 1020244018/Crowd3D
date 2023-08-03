import torch
import numpy as np

from .muco import MuCo
from .agora import AGORA
from .h36m import H36M
from .largecrowd import LargeCrowd
from .test_train_scale import TEST_TRAIN_SCALE
from .eval_largecrowd import EVAL_LargeCrowd
from .test_train_single_image import TEST_TRAIN_SINGLE_IMAGE
from .single_scene_image import SINGLE_SCENE_IMAGE


import sys, os
from prettytable import PrettyTable

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from dataset.image_base import *
import config
from config import args
from collections import OrderedDict

dataset_dict = {'h36m': H36M, 'muco': MuCo, 'agora': AGORA,'largecrowd': LargeCrowd,
                'test_train_scale': TEST_TRAIN_SCALE, 'test_train_single_image': TEST_TRAIN_SINGLE_IMAGE, 
                'eval_largecrowd': EVAL_LargeCrowd, 'single_scene_image': SINGLE_SCENE_IMAGE
                }


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        datasets_used = args().dataset.split(',')
        self.datasets = [dataset_dict[ds](**kwargs) for ds in datasets_used]

        self.lengths, self.partition, self.ID_num_list, self.ID_num = [], [], [], 0
        sample_prob_dict = args().sample_prob_dict
        for ds_idx, ds_name in enumerate(datasets_used):
            self.lengths.append(len(self.datasets[ds_idx]))
            self.partition.append(sample_prob_dict[ds_name])
            if self.datasets[ds_idx].ID_num > 0:
                self.ID_num_list.append(self.ID_num)
                self.ID_num += self.datasets[ds_idx].ID_num
            else:
                self.ID_num_list.append(0)
        dataset_info_table = PrettyTable([' '] + datasets_used)
        dataset_info_table.add_row(['Length'] + self.lengths)
        dataset_info_table.add_row(['Sample Prob.'] + self.partition)
        expect_length = (np.array(self.lengths) / np.array(self.partition)).astype(np.int)
        dataset_info_table.add_row(['Expected length'] + expect_length.tolist())
        self.partition = np.array(self.partition).cumsum()
        dataset_info_table.add_row(['Accum. Prob.'] + self.partition.astype(np.float16).tolist())
        dataset_info_table.add_row(['Accum. ID.'] + self.ID_num_list)
        print(dataset_info_table)
        self.total_length = int(expect_length.max())
        logging.info('All dataset length: {}'.format(len(self)))

    def _get_ID_num_(self):
        return self.ID_num

    def __getitem__(self, index):
        p = float(index) / float(self.total_length)
        for inds in range(len(self.partition)):
            if p <= self.partition[inds]:
                pre_prob = self.partition[inds - 1] if inds > 0 else 0
                sample_prob = (p - pre_prob) / (self.partition[inds] - pre_prob)
                omit_internal = self.lengths[inds] // ((self.partition[inds] - pre_prob) * self.total_length)
                index_sample = int(
                    min(self.lengths[inds] * sample_prob + random.randint(0, omit_internal), self.lengths[inds] - 1))
                annots = self.datasets[inds][index_sample]
                annots['subject_ids'] += self.ID_num_list[inds]
                return annots

    def __len__(self):
        return self.total_length


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, **kwargs):
        assert dataset in dataset_dict, print('dataset {} not found while creating data loader!'.format(dataset))
        self.dataset = dataset_dict[dataset](**kwargs)
        self.length = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    config.datasets_used = ['pw3d', 'crowdpose', 'posetrack', 'oh']
    datasets = MixedDataset(train_flag=True)
    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset=datasets, batch_size=64, shuffle=True, drop_last=True, pin_memory=True,
                             num_workers=1)
    for data in enumerate(data_loader):
        pass
