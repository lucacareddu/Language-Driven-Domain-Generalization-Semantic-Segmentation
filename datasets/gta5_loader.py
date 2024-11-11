import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class GTA5Dataset(Dataset):
    def __init__(self, root, ignore_index, resize=None, transforms=None, rcs=False, rcs_temp=0.01, stats=False):
        self.root = root
        self.ignore_index = ignore_index
        self.resize = resize
        self.transforms = transforms
        self.stats = stats

        self.gtaid2cityid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                             19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                             26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        
        self.files = {
            "images" : glob.glob(f"{os.path.join(root,'images')}/*.png"),
            "labels" : glob.glob(f"{os.path.join(root,'labels')}/*.png")
        }

        assert len(self.files["images"]) == len(self.files["labels"]) == 24966

        # self.rand_indices = np.random.choice(list(range(len(self.files["images"]))), size=500, replace=False)

        self.rcs = rcs
        if rcs:
            self.class_rareness, self.class_popul, self.class_centr = self.get_rcs_stats(rcs_temp)

            # from utils.colors import CITY_VALID_CLASSES

            # for i, (n,r,p) in enumerate(zip(CITY_VALID_CLASSES, self.class_rareness, self.class_popul)):
            #     print('Class {} {} Rar {:.3f} Ex {}'.format(i, n, r, len(p)))


    def __len__(self):
        return len(self.files["images"]) # len(self.rand_indices)


    def __getitem__(self, idx):
        # idx = self.rand_indices[idx]
        if self.rcs:
            c = np.random.choice(list(range(19)), p=self.class_rareness)
            f = self.class_popul[c].pop()
            idx = self.files["images"].index(os.path.join(self.root,"images",f))
            
        img = Image.open(self.files["images"][idx]).convert('RGB')
        lbl = Image.open(self.files["labels"][idx])
        name = self.files["images"][idx].split("/")[-1]

        if self.resize:
            img = img.resize(self.resize, Image.BICUBIC)
            lbl = lbl.resize(self.resize, Image.NEAREST)

        if self.transforms:
            img, lbl = self.transforms(img, lbl) if not self.rcs else self.transforms(img, lbl, self.class_centr[c].pop())
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
        if self.rcs:
            output["rare_class"] = torch.tensor(c)

        return output
    

    def encode_label(self, lbl):
        # re-assign labels to match the format of Cityscapes
        label = self.ignore_index * np.ones(lbl.shape, dtype=np.uint8)
        for k, v in self.gtaid2cityid.items():
            label[lbl == k] = v
        return label
    

    def get_rcs_stats(self, temperature):
        with open(os.path.join(self.root, 'GTA5_class_pixels.json'), 'r') as of:
            class_pixels_freq = json.load(of)

        freq = torch.tensor(list(class_pixels_freq.values()))
        freq = freq / torch.sum(freq)
        freq = 1 - freq
        freq = torch.softmax(freq / temperature, dim=-1)

        with open(os.path.join(self.root, 'GTA5_class_popul.json'), 'r') as of:
            class_popul_names = json.load(of)
            class_popul_names = list(class_popul_names.values())

        with open(os.path.join(self.root, 'GTA5_class_centr.json'), 'r') as of:
            class_popul_centr = json.load(of)
            class_popul_centr = list(class_popul_centr.values())

        return freq.numpy(), class_popul_names, class_popul_centr



if __name__=="__main__":
    import sys
    sys.path.insert(0,"/home/luca/work")
    from torch.utils.data import DataLoader
    from utils.visualization import visualize
    from transformscpu import *
    from transformsgpu import train_aug
    from utils.colors import GTA_MEAN, GTA_STD

    transforms = Compose([CentroidCCrop((512,512))])

    ds = GTA5Dataset(root="/home/luca/data/gta", ignore_index=255, resize=(1280,720), transforms=transforms, rcs=True)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1)

    for b in dl:
        images, labels = b["img"], b["lbl"]
        visualize(img=images.squeeze(0).numpy(), lbl=labels.squeeze(0).numpy(), pred=labels.squeeze(0).numpy(), fullscreen=False)

        images, labels = train_aug(images=images, labels=labels, normalization=False)
        visualize(images.squeeze(0).numpy(), labels.squeeze(0).numpy())

        images, labels = train_aug(images=images, labels=labels, mean=GTA_MEAN, std=GTA_STD)
        visualize(images.squeeze(0).numpy(), labels.squeeze(0).numpy())