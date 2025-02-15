import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MapillaryVistasVal(Dataset):
    def __init__(self, root, ignore_index, resize=None, transforms=None, stats=False):
        self.root = root
        self.ignore_index = ignore_index
        self.resize = resize
        self.transforms = transforms
        self.stats = stats
        
        self.vistasid2cityid = {13: 0, 24: 0, 41: 0, 2: 1, 15: 1, 17: 2, 6: 3, 3: 4, 45: 5, 47: 5, 48: 6, 50: 7, 30: 8, 29: 9, 27: 10, 19: 11, 20: 12, 21: 12, 22: 12, 55: 13, 61: 14, 54: 15, 58: 16, 57: 17, 52: 18}

        self.files = {
            "images" : glob.glob(f"{os.path.join(root, 'images')}/*.jpg"),
            "labels" : glob.glob(f"{os.path.join(root, 'v1.2/labels')}/*.png")
        }

        self.files["images"].sort()
        self.files["labels"].sort()

        assert len(self.files["images"]) == len(self.files["labels"]) == 2000


    def __len__(self):
        return len(self.files["images"])


    def __getitem__(self, idx):
        img = Image.open(self.files["images"][idx]).convert('RGB')
        lbl = Image.open(self.files["labels"][idx])
        name = self.files["images"][idx].split("/")[-1]

        if self.resize:
            img = img.resize(self.resize, Image.BICUBIC)
            lbl = lbl.resize(self.resize, Image.NEAREST)

        if self.transforms:
            img, lbl = self.transforms(img, lbl)
        else:
            img = np.array(img, np.float32) / 255
            lbl = np.array(lbl, np.uint8)

        lbl = self.encode_label(lbl)

        img = torch.from_numpy(np.transpose(img, (2,1,0)))
        lbl = torch.from_numpy(np.transpose(lbl)).long()

        if self.stats:
            output = {"img": img, "lbl": lbl, "fname": name}
            return output

        classes = torch.unique(lbl)
        classes = classes[classes != self.ignore_index]

        binary_masks = lbl.repeat(len(classes),1,1)
        binary_masks = (binary_masks == classes[:,None,None]).float()

        output = {"img": img, 
                "lbl": lbl,
                "classes" : classes, 
                "bin_masks": binary_masks,
                "fname": name
                }

        return output
    

    def encode_label(self, lbl):
        # re-assign labels to match the format of Cityscapes
        label = self.ignore_index * np.ones(lbl.shape, dtype=np.uint8)
        for k, v in self.vistasid2cityid.items():
            label[lbl == k] = v
        return label