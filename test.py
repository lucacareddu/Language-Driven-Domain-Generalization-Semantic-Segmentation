import os
import datetime
timestamp = datetime.datetime.now().strftime('%d-%m_%H-%M-%S')

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import DGSSModel

from datasets import GTA5Dataset, CityscapesDataset
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
use_classes = config["decoder"]["use_classes"]
predict_classes = config["decoder"]["predict_classes"]

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

gta_augmentations = Compose([RandomCrop(crop_size)])
gta_val_indices = np.random.choice(24966, size=500, replace=False)

train_gta = GTA5Dataset(root=gta_root, split="train", val_indices=gta_val_indices, ignore_index=ignore_index, resize=gta_inp_size, transforms=gta_augmentations)
val_gta = GTA5Dataset(root=gta_root, split="val", val_indices=gta_val_indices, ignore_index=ignore_index, resize=gta_inp_size)
val_city = CityscapesDataset(root=city_root, split="val", ignore_index=ignore_index, resize=city_inp_size)

gta_train_loader = DataLoader(train_gta, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True,  worker_init_fn=lambda id: fix_seed(id), collate_fn=collate_fn)
gta_val_loader = DataLoader(val_gta, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
city_val_loader = DataLoader(val_city, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

#################################################################################################

text_prompts = None

if use_text:
    if classdef_prompts:
        print("Class definitions employed.")
        with open("class_definition/class_definition.json","r") as f:
            class_definition = json.load(f)
            class_definition = df_dict_search(dictionary=class_definition, class_names=CITY_VALID_CLASSES)
            text_prompts = [f"{c}: " + class_definition[c] for c in CITY_VALID_CLASSES]
            # print([len(x) for x in text_prompts])
    else:
        print("Class names employed.")
        text_prompts = [f"a photo of a {c}." for c in CITY_VALID_CLASSES]

#################################################################################################

model = DGSSModel(encoder_name=encoder_name, ignore_value=ignore_index, text_prompts=text_prompts, freeze_vision_encoder=freeze_vision, freeze_text_encoder=freeze_text, shallow_m2f=shallow_m2f, use_classes=use_classes, predict_classes=predict_classes)
model.to(device)

model.print_trainable_params()
model.print_frozen_modules()

params = []

if not freeze_vision:
    if "clip" in encoder_name and freeze_text:
        params.append({'name':"encoder", 'params': model.encoder.vision_model.parameters()})
    else:
        params.append({'name':"encoder", 'params': model.encoder.parameters()})
        if encoder_name == "vit" and model.has_text_decoder and not freeze_text:
            params.append({'name':"encoder", 'params': model.vit_text_encoder.parameters()})

params.append({'name':"neck", 'params': model.neck.parameters()})
params.append({'name':"vision_decoder", 'params': model.vision_decoder.parameters()})

if model.has_text_decoder:
    params.append({'name':"text_decoder", 'params': model.text_decoder.parameters()})
    
optimizer = torch.optim.AdamW(params, lr=lr)

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

path = "checkpoints/12-01_14-43-33"

files = sorted(glob.glob(f"{path}/*.pth"), key = lambda x: int(f3(x)[-1]))


for resume_path in files:
    i_iter = resume_checkpoint(resume_path, model, optimizer) - 1

    print()
    print(path, " : ", i_iter)
    print()

    model.eval()

    for val_name, val_loader, stride in zip(["gta", "city"], [gta_val_loader, city_val_loader], [(426,426), (341,341)]):
        if val_name == "gta":
            continue
        with torch.no_grad():
            runn_loss = torch.zeros((1)).to(device)
            runn_bins = torch.zeros((3, 19)).to(device)
            loop = tqdm(val_loader, leave=False)
            
            for i,batch in enumerate(loop):
                if i==1:
                    break
                images = batch["image"].to(device)
                labels = batch["label"].to(device)              

                if normalization:
                    images = normalize(images)
                
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
                        
                        loss, cl_loss, cl_acc, crop_upsampled_logits = model(pixel_values=crop_img, bin_masks=crop_binmasks, classes=crop_classes, return_logits=True)
                        
                        preds += torch.nn.functional.pad(crop_upsampled_logits, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                        count_mat[:, :, y1:y2, x1:x2] += 1

                assert (count_mat == 0).sum() == 0
                preds = preds / count_mat

                upsampled_logits = preds.argmax(dim=1).detach()

                runn_loss.add_(loss)
                runn_bins.add_(get_confBins(predictions=upsampled_logits, references=labels, ignore_index=ignore_index))
            
            mloss = runn_loss.item() / len(val_loader)
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
                tb_writer.add_scalar(f"CL_Loss ({val_name}_val): ", cl_loss, i_iter)
                tb_writer.add_scalar(f"CL_mAcc ({val_name}_val): ", cl_acc, i_iter)
