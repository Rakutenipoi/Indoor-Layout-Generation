import numpy as np
from .process_dataset import *

class COFSDataset(Dataset):
    def __init__(self, data, config):
        self.config = config
        self.centroids = data['floor_plan_centroid'].values
        self.max_object_tokens = config['object_max_num'] * config['attributes_num']
        self.max_len = (config['object_max_num'] + 2) * config['attributes_num']
        self.max_object = config['object_max_num']
        # bounding box
        self.bounds_translations = np.array(config['bounds_translations'])
        self.bounds_sizes = np.array(config['bounds_sizes'])
        self.bounds_angles = np.array(config['bounds_angles'])
        # data
        self.layout = data['room_layout'].values
        self.class_labels = data['class_labels'].values
        self.translations = data['translations'].values
        self.sizes = data['sizes'].values
        self.angles = data['angles'].values
        # special token
        self.pad = np.array([config['padding_token']] + [0.] * (config['attributes_num'] - 1))
        self.bos = np.array([config['start_token']] + [0.] * (config['attributes_num'] - 1))
        self.eos = np.array([config['end_token']] + [0.] * (config['attributes_num'] - 1))

        self.normalize()

    def normalize(self):
        for i in range(len(self.translations)):
            self.translations[i] = (self.translations[i] - self.bounds_translations[:3]) / (self.bounds_translations[3:] - self.bounds_translations[:3])
            self.sizes[i] = (self.sizes[i] - self.bounds_sizes[:3]) / (self.bounds_sizes[3:] - self.bounds_sizes[:3])
            self.angles[i] = (self.angles[i] - self.bounds_angles[0]) / (self.bounds_angles[1] - self.bounds_angles[0])

    def __len__(self):
        return len(self.layout)

    def __getitem__(self, idx):
        layout = self.layout[idx]
        class_labels = np.argmax(self.class_labels[idx], axis=-1)
        translations = self.translations[idx]
        sizes = self.sizes[idx]
        angles = self.angles[idx]

        sequence_length = class_labels.shape[0]
        sequence = [
            [class_label, trans[0], trans[1], trans[2], size[0], size[1], size[2], angle[0]]
            for class_label, trans, size, angle in zip(class_labels, translations, sizes, angles)
        ]
        if sequence_length > self.max_object:
            sequence = sequence[:self.max_object]
            sequence_length = self.max_object
        sequence = np.reshape(sequence, (-1))

        # src
        src = np.concatenate([self.bos, sequence, self.eos])
        src_len = sequence_length + 2
        # tgt
        tgt = np.concatenate([self.bos, sequence])
        tgt_len = sequence_length + 1
        # tgt_y
        tgt_y = np.concatenate([sequence, self.eos])
        tgt_y_len = sequence_length + 1

        # padding
        if src_len < self.max_object + 2:
            pad = np.array([self.config['padding_token']] * (self.max_len - src_len * self.config['attributes_num']))
            src = np.concatenate((src, pad), axis=0)
        src = torch.tensor(src, dtype=torch.float32)
        if tgt_len < self.max_object + 2:
            pad = np.array([self.config['padding_token']] * (self.max_len - tgt_len * self.config['attributes_num']))
            tgt = np.concatenate((tgt, pad), axis=0)
        tgt = torch.tensor(tgt, dtype=torch.float32)
        if tgt_y_len < self.max_object + 2:
            pad = np.array([self.config['padding_token']] * (self.max_len - tgt_y_len * self.config['attributes_num']))
            tgt_y = np.concatenate((tgt_y, pad), axis=0)
        tgt_y = torch.tensor(tgt_y, dtype=torch.float32)

        # layout normalization
        layout = layout / 255.0

        return layout, src, tgt, tgt_y, src_len









