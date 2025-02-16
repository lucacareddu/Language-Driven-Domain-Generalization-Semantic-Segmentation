import os
import datetime
timestamp = datetime.datetime.now().strftime('%d-%m_%H-%M-%S')

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import DGSSModel

from datasets import GTA5Dataset, CityscapesDataset
from datasets import ACDCVal, MapillaryVistasVal
from datasets.transformscpu import *

from torchvision import transforms
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

from torch.utils.data import DataLoader

from utils import *

from tqdm import tqdm, trange
from torch.utils import tensorboard
import json


#################################################################################################

resume_path = None

if resume_path is not None:
    config = json.load(open(os.path.join('/'.join(resume_path.split("/")[:-1]), "config.json")))
    print("Resuming from: ", resume_path)
else:
    config = json.load(open("configs/config.json"))

SEED = config["SEED"]
debug = config["debug_mode"]

encoder_name = config["encoder"]["name"]
freeze_vision = config["encoder"]["freeze_vision"]
freeze_text = config["encoder"]["freeze_text"]

shallow_m2f = config["decoder"]["shallow_m2f"]
use_text = config["decoder"]["use_text"]
classdef_prompts = config["decoder"]["classdef_prompts"]

gta_root = config["gta"]["remote_root"] if config["remote"] else config["gta"]["local_root"]
gta_inp_size = tuple(config["gta"]["input_size"])
city_root = config["city"]["remote_root"] if config["remote"] else config["city"]["local_root"]
city_inp_size = tuple(config["city"]["input_size"])
crop_size = tuple(config["preprocessing"]["crop_size"])
ignore_index = config["preprocessing"]["ignore_index"]
normalization = config["preprocessing"]["normalize"]

batch_size = config["training"]["batch_size"]
num_workers = config["training"]["num_workers"]
max_iterations = config["training"]["max_iterations"]
iters_per_val = config["training"]["iters_per_val"]
log_dir = config["training"]["log_dir"]
do_checkpoints = config["training"]["do_checkpoints"]
iters_per_save = config["training"]["iters_per_save"]
checkpoint_dir = config["training"]["checkpoint_dir"]

grad_clip = config["grad_clip"]["enable"]
grad_clip_value = config["grad_clip"]["small_model"] if encoder_name == "tiny_clip" else config["grad_clip"]["large_model"]
lr = config["optimizer"]["learning_rate"]
lr_power = config["optimizer"]["lr_power"]
lr_warmup_iters = config["optimizer"]["lr_warmup_iterations"]

#################################################################################################

if True:
    fix_seed(SEED=SEED)

#################################################################################################

val_city = CityscapesDataset(root=city_root, split="val", ignore_index=ignore_index, resize=city_inp_size, stats=True)
val_acdc_night = ACDCVal(root="/home/thesis/datasets/ACDC", split="night", ignore_index=ignore_index, resize=city_inp_size, stats=True)
val_acdc_rain = ACDCVal(root="/home/thesis/datasets/ACDC", split="rain", ignore_index=ignore_index, resize=city_inp_size, stats=True)
val_acdc_fog = ACDCVal(root="/home/thesis/datasets/ACDC", split="fog", ignore_index=ignore_index, resize=city_inp_size, stats=True)
val_acdc_snow = ACDCVal(root="/home/thesis/datasets/ACDC", split="snow", ignore_index=ignore_index, resize=city_inp_size, stats=True)
val_acdc_all = ACDCVal(root="/home/thesis/datasets/ACDC", split="all", ignore_index=ignore_index, resize=city_inp_size, stats=True)
val_vistas = MapillaryVistasVal(root="/home/thesis/datasets/mapi_val", ignore_index=ignore_index, resize=city_inp_size, stats=True)

city_val_loader = DataLoader(val_city, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
acdc_night_val_loader = DataLoader(val_acdc_night, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
acdc_rain_val_loader = DataLoader(val_acdc_rain, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
acdc_fog_val_loader = DataLoader(val_acdc_fog, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
acdc_snow_val_loader = DataLoader(val_acdc_snow, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
acdc_all_val_loader = DataLoader(val_acdc_all, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
vistas_val_loader = DataLoader(val_vistas, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

val_names = ["city", "acdc_night", "acdc_rain", "acdc_fog", "acdc_snow", "acdc_all", "vistas"]
val_loaders = [city_val_loader, acdc_night_val_loader, acdc_rain_val_loader, acdc_fog_val_loader, acdc_snow_val_loader, acdc_all_val_loader, vistas_val_loader]

#################################################################################################

text_prompts = None

if use_text:
    if classdef_prompts:
        print("Class definitions employed.")
        with open("utils/class_definition/class_definition.json","r") as f:
            class_definition = json.load(f)
            class_definition = df_dict_search(dictionary=class_definition, class_names=CITY_VALID_CLASSES)
            text_prompts = [f"{c}: " + class_definition[c] for c in CITY_VALID_CLASSES]
            # print([len(x) for x in text_prompts])
    else:
        print("Class names employed.")
        animals = ["bird", "cat", "dog", "ferret", "fish", "gerbil", "guinea pig", "hamster", "lizard", "mouse", "rabbit", "rat", "snake", "turtle", "alpaca", "cow", "chicken", "donkey", "goat"]
        text_prompts = [f"a photo of a {c}." for c in CITY_VALID_CLASSES]

#################################################################################################

model = DGSSModel(encoder_name=encoder_name, ignore_value=ignore_index, text_prompts=text_prompts, freeze_vision_encoder=freeze_vision, freeze_text_encoder=freeze_text, shallow_m2f=shallow_m2f)
model.to(device)

model.print_trainable_params()
model.print_frozen_modules()
    
optimizer = None

#################################################################################################

if not debug:
    log_dir = os.path.join(log_dir, timestamp)
    os.makedirs(log_dir)
    tb_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

#################################################################################################

import re
import glob

def f3(string):
    return re.findall(r'[0-9]+', string)

checkpoint_path = {"clip_lang": "saved/27-01_11-04-47", "clip_rand": "saved/30-01_12-48-49", "vit_lang": "checkpoints/10-02_12-23-34", "vit_rand": "checkpoints/11-02_14-22-41"}
path = checkpoint_path["clip_lang"]

files = sorted(glob.glob(f"{path}/*.pth"), key = lambda x: int(f3(x)[-1]))


for resume_path in files:
    i_iter = resume_checkpoint(resume_path, model, optimizer) - 1

    print()
    print(path, " : ", i_iter)
    print()

    model.eval()

    for val_name, val_loader, stride in zip(val_names, val_loaders, [(341,341)]*len(val_loaders)):
        with torch.no_grad():
            runn_loss = torch.zeros((1)).to(device)
            runn_bins = torch.zeros((3, 19)).to(device)
            runn_aux_loss = torch.zeros((1)).to(device)
            runn_aux_miou = torch.zeros((1)).to(device)
            loop = tqdm(val_loader, leave=False)
            
            for batch in loop:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)              

                if normalization:
                    images = normalize(images)

                # SLIDE-INFERENCE adapted from DenseCLIP official MMSeg implementation
                
                h_stride, w_stride = stride
                h_crop, w_crop = crop_size
                batch_size, _, h_img, w_img = images.size()
                num_classes = 19
                h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
                preds = images.new_zeros((batch_size, num_classes, h_img, w_img))
                count_mat = images.new_zeros((batch_size, 1, h_img, w_img))

                for h_idx in range(h_grids):
                    for w_idx in range(w_grids):
                        y1 = h_idx * h_stride
                        x1 = w_idx * w_stride
                        y2 = min(y1 + h_crop, h_img)
                        x2 = min(x1 + w_crop, w_img)
                        y1 = max(y2 - h_crop, 0)
                        x1 = max(x2 - w_crop, 0)

                        crop_img = images[:, :, y1:y2, x1:x2]
                        crop_labels = labels[:, y1:y2, x1:x2]
                        crop_classes = [torch.unique(x) for x in crop_labels]
                        crop_classes = [x[x != ignore_index] for x in crop_classes]
                        crop_binmasks = [(l.repeat(len(c),1,1) == c[:,None,None]).float() for l,c in zip(crop_labels, crop_classes)]
                        
                        outs = model(pixel_values=crop_img, bin_masks=crop_binmasks, classes=crop_classes, labels=crop_labels, return_logits=True)
                        
                        preds += torch.nn.functional.pad(outs["upsampled_logits"], (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                        count_mat[:, :, y1:y2, x1:x2] += 1

                        runn_loss.add_(outs["loss"] / (h_grids*w_grids))
                        if "aux_loss" in outs.keys():
                            runn_aux_loss.add_(outs["aux_loss"] / (h_grids*w_grids))
                        if "aux_miou" in outs.keys():
                            runn_aux_miou.add_(outs["aux_miou"] / (h_grids*w_grids))

                assert (count_mat == 0).sum() == 0
                preds = preds / count_mat

                upsampled_logits = preds.argmax(dim=1).detach()

                runn_bins.add_(get_confBins(predictions=upsampled_logits, references=labels, ignore_index=ignore_index))
            
            mloss = runn_loss.item() / len(val_loader)
            aux_mloss = runn_aux_loss.item() / len(val_loader)
            aux_miou = runn_aux_miou.item() / len(val_loader)
            jaccard, accuracy = get_metrics(runn_bins)
            miou = torch.nanmean(jaccard).item()
            macc = torch.nanmean(accuracy).item()

            perclass_repr(torch.stack((jaccard, accuracy)).cpu().numpy().transpose())
            print(f"Loss ({val_name}_val): ", mloss)
            print(f"mIoU ({val_name}_val): ", miou)
            print(f"mAcc ({val_name}_val): ", macc)
            
            if not debug:
                tb_writer.add_scalar(f"Loss ({val_name}_val): ", mloss, i_iter)
                tb_writer.add_scalar(f"mIoU ({val_name}_val): ", miou, i_iter)
                tb_writer.add_scalar(f"mAcc ({val_name}_val): ", macc, i_iter)
                if "aux_loss" in outs.keys():
                    tb_writer.add_scalar(f"aux_loss ({val_name}_val): ", aux_mloss, i_iter)
                if "aux_miou" in outs.keys():
                    tb_writer.add_scalar(f"aux_mIoU ({val_name}_val): ", aux_miou, i_iter)
