import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ACDCVal(Dataset):
    def __init__(self, root, split, ignore_index, resize=None, transforms=None, stats=False):
        self.root = root
        self.split = split
        self.ignore_index = ignore_index
        self.resize = resize
        self.transforms = transforms
        self.stats = stats

        if split == "all":
            self.files = {
                "images" : glob.glob(f"{os.path.join(root, 'rgb_anon/night/val')}/*/*.png") 
                            + glob.glob(f"{os.path.join(root, 'rgb_anon/rain/val')}/*/*.png")
                            + glob.glob(f"{os.path.join(root, 'rgb_anon/fog/val')}/*/*.png")
                            + glob.glob(f"{os.path.join(root, 'rgb_anon/snow/val')}/*/*.png"),
                "labels" : glob.glob(f"{os.path.join(root, 'gt/night/val')}/*/*_labelTrainIds.png") 
                            + glob.glob(f"{os.path.join(root, 'gt/rain/val')}/*/*_labelTrainIds.png")
                            + glob.glob(f"{os.path.join(root, 'gt/fog/val')}/*/*_labelTrainIds.png")
                            + glob.glob(f"{os.path.join(root, 'gt/snow/val')}/*/*_labelTrainIds.png"),
            }
        else:
            self.files = {
                "images" : glob.glob(f"{os.path.join(root, f'rgb_anon/{split}/val')}/*/*.png"),
                "labels" : glob.glob(f"{os.path.join(root, f'gt/{split}/val')}/*/*_labelTrainIds.png"),
            }

        self.files["images"].sort()
        self.files["labels"].sort()

        split_lenghts = {"all":406, "night":106, "rain":100, "fog":100, "snow":100}

        assert len(self.files["images"]) == len(self.files["labels"]) == split_lenghts[split]


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

        lbl[lbl >= 19] = self.ignore_index

        img = torch.from_numpy(np.transpose(img, (2,1,0)))
        lbl = torch.from_numpy(np.transpose(lbl)).long()

        if self.stats:
            output = {"img": img, "lbl": lbl, "fname": name}
            return output

        classes = torch.unique(lbl)
        print(classes)
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
