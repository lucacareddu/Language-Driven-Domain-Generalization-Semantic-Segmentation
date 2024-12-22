import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CityscapesDataset(Dataset):
    def __init__(self, root, split, ignore_index, resize=None, transforms=None, stats=False):
        self.root = root
        self.split = split
        self.ignore_index = ignore_index
        self.resize = resize
        self.transforms = transforms
        self.stats = stats
        
        images_prefix = 'leftImg8bit_trainvaltest/leftImg8bit' # local folder
        labels_prefix = 'gtFine_trainvaltest/gtFine' # local folder
        
        images_prefix = images_prefix if os.path.exists(os.path.join(root, images_prefix)) else 'rgb' # remote folder
        labels_prefix = labels_prefix if os.path.exists(os.path.join(root, labels_prefix)) else 'gt' # remote folder
        
        self.files = {
            "images" : glob.glob(f"{os.path.join(root, images_prefix, self.split)}/*/*.png"),
            "labels" : glob.glob(f"{os.path.join(root, labels_prefix, self.split)}/*/*gtFine_labelIds.png")
        }
        
        self.files["images"].sort()
        self.files["labels"].sort()

        split_size = {"train":2975, "val":500, "test":1525}
        assert len(self.files["images"]) == len(self.files["labels"]) == split_size[split]

        # self.rand_indices = np.random.choice(list(range(len(self.files["images"]))), size=500, replace=False)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.valid_map = dict(zip(self.valid_classes, range(19)))


    def __len__(self):
        return len(self.files["images"]) # len(self.rand_indices)
    

    def __getitem__(self, idx):
        # idx = self.rand_indices[idx]
        
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

        if 1: 
            # Training
            img = torch.from_numpy(np.transpose(img,(2, 1, 0)))
            lbl = torch.from_numpy(np.transpose(lbl)).long()

            if self.stats or self.split == "val":
                output = {"img": img, "lbl": lbl, "fname": name}
                return output

            classes = torch.unique(lbl)
            classes = classes[classes != self.ignore_index]

            binary_masks = lbl.repeat(len(classes),1,1)
            binary_masks = (binary_masks == classes[:,None,None]).float()
        else: 
            # Evaluation
            img = torch.stack([torch.from_numpy(np.transpose(x,(2, 1, 0))) for x in img])
            lbl = torch.stack([torch.from_numpy(np.transpose(x)) for x in lbl]).long()

            classes = [torch.unique(x) for x in lbl]
            classes = [x[x != self.ignore_index] for x in classes]

            binary_masks = [(lbl_x.repeat(len(classes_x),1,1) == classes_x[:,None,None]).float() for classes_x, lbl_x in zip(classes, lbl)]
        
        output = {"img": img, 
                "lbl": lbl,
                "classes" : classes, 
                "bin_masks": binary_masks,
                "fname": name
                }

        return output
    

    def encode_label(self, lbl):
        label = self.ignore_index * np.ones(lbl.shape, dtype=np.uint8)
        for k, v in self.valid_map.items():
            label[lbl == k] = v
        return label
    


if __name__=="__main__":
    import sys
    sys.path.insert(0,"/home/luca/work")
    from torch.utils.data import DataLoader
    from utils.visualization import visualize
    from transformscpu import *
    from transformsgpu import eval_aug
    from utils.colors import CITY_MEAN, CITY_STD

    transforms = Compose([TwoCropsCityVal((512,512))])

    ds = CityscapesDataset(root="/home/luca/data/cityscapes", split="val", ignore_index=255, resize=(1024,512), transforms=transforms)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1)

    for b in dl:
        img1, lbl1 = b["img"][0,0], b["lbl"][0,0]
        visualize(img=img1.numpy(), lbl=lbl1.numpy(), fullscreen=True)
        img2, lbl2 = b["img"][:,1], b["lbl"][:,1]
        visualize(img=img2.squeeze(0).numpy(), lbl=lbl2.squeeze(0).numpy(), fullscreen=True)

        images, labels = eval_aug(images=img2, labels=lbl2, mean=CITY_MEAN, std=CITY_STD)
        visualize(images.squeeze(0).numpy(), labels.squeeze(0).numpy())