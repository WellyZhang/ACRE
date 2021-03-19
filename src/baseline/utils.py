# -*- coding: utf-8 -*-


import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ACREDataset(Dataset):
    def __init__(self, dataset_path, split, img_size, num_digits_questions=6, num_digits_panels=2):
        self.dataset_path = dataset_path
        self.img_size = img_size

        prefix = "ACRE_%s" % split
        self.img_template = "%s_%%0%dd_%%0%dd.png" % (prefix, num_digits_questions, num_digits_panels)
        self.img_template = os.path.join(self.dataset_path, "images", self.img_template)
        config_path = os.path.join(self.dataset_path, "config", "{}.json".format(split))
        with open(config_path, "r") as f:
            self.config = json.load(f)
        q_types = ["direct", "indirect", "screen_off", "potential"]
        self.q_type_encodings = dict()
        for i in range(len(q_types)):
            encoding = [0] * len(q_types)
            encoding[i] = 1
            self.q_type_encodings[q_types[i]] = encoding

    def __len__(self):
        return len(self.config)

    def __getitem__(self, idx):
        image = []
        target = []
        q_type = []
        for i in range(10):
            img = Image.open(self.img_template % (idx, i))
            width, height = img.size
            if not (width == self.img_size[0] and height == self.img_size[1]):
                img = img.resize(self.img_size)
            img = np.array(img)
            image.append(img[:, :, :-1])
            if i >= 6:
                target.append(self.config[idx][i]["label"])
                q_type.append(self.q_type_encodings[self.config[idx][i]["type"]])
        
        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(0, 3, 1, 2)
        target = torch.tensor(target, dtype=torch.long)
        q_type = torch.tensor(q_type, dtype=torch.int)

        return image, target, q_type
