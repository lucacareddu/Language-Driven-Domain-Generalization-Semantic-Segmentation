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


debug = config["debug_mode"]

encoder_name = config["encoder"]["name"]
use_text = "clip" in encoder_name and config["encoder"]["use_text"]
gta_root = config["gta"]["remote_root"] if config["remote"] else config["gta"]["local_root"]
gta_inp_size = tuple(config["gta"]["input_size"])
city_root = config["city"]["remote_root"] if config["remote"] else config["city"]["local_root"]
city_inp_size = tuple(config["city"]["input_size"])
crop_size = tuple(config["preprocessing"]["crop_size"])
ignore_index = config["preprocessing"]["ignore_index"]
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

SEED = 0

if True:
    fix_seed(SEED=SEED)

#################################################################################################

gta_augmentations = Compose([CentroidCCrop(crop_size)])
city_val_augmentations = Compose([TwoCropsCityVal(crop_size)])

gta_val_indices = np.random.choice(24966, size=1, replace=False)

train_gta = GTA5Dataset(root=gta_root, split="train", val_indices=gta_val_indices, ignore_index=ignore_index, resize=gta_inp_size, transforms=gta_augmentations)
val_gta = GTA5Dataset(root=gta_root, split="val", val_indices=gta_val_indices, ignore_index=ignore_index, resize=gta_inp_size)
val_city = CityscapesDataset(root=city_root, split="val", ignore_index=ignore_index, resize=city_inp_size)

gta_train_loader = DataLoader(train_gta, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True, collate_fn=collate_fn)
gta_val_loader = DataLoader(val_gta, batch_size=batch_size//2, num_workers=num_workers, collate_fn=collate_fn)
city_val_loader = DataLoader(val_city, batch_size=batch_size//2, num_workers=num_workers, collate_fn=collate_fn)

text_prompts = None

if use_text:
    text_prompts = [f"a photo of a {c}." for c in CITY_VALID_CLASSES]

#################################################################################################

model = DGSSModel(encoder_name=encoder_name, ignore_value=ignore_index, text_prompts=text_prompts)
model.to(device)

model.print_trainable_params()
model.print_frozen_modules()

params = []

if "clip" in model.encoder_name and model.freeze_text_encoder:
    params.append({'name':"encoder", 'params': model.encoder.vision_model.parameters()})
else:
    params.append({'name':"encoder", 'params': model.encoder.parameters()})

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

path = "checkpoints/25-11_11-24-21"

files = sorted(glob.glob(f"{path}/*.pth"), key = lambda x: int(f3(x)[-1]))

# print(torch.mean(model.missing_class_predictor.weight), torch.std(model.missing_class_predictor.weight))

for resume_path in files:
    i_iter = resume_checkpoint(resume_path, model, optimizer) - 1

    print()
    print(path, " : ", i_iter)
    print()

    # print(torch.mean(model.missing_class_predictor.weight), torch.std(model.missing_class_predictor.weight))
    
    model.eval()
    for val_name, val_loader in zip(["gta", "city"], [gta_val_loader, city_val_loader]):
            with torch.no_grad():
                runn_loss = torch.zeros((1)).to(device)
                runn_bins = torch.zeros((3, 19)).to(device)
                loop = tqdm(val_loader, leave=False)
                
                for batch in loop:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    # classes = [x.to(device) for x in batch["classes"]]
                    # binmasks = [x.to(device) for x in batch["bin_masks"]]                

                    if 1:
                        images = normalize(images)
                    
                    h_stride, w_stride = (341,341)
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
                            crop_classes = [c[c != ignore_index] for c in crop_classes]
                            crop_binmasks = [(l.repeat(len(c),1,1) == c[:,None,None]).float() for l,c in zip(crop_labels, crop_classes)]
                            loss, crop_upsampled_logits = model(pixel_values=crop_img, bin_masks=crop_binmasks, classes=crop_classes, return_logits=True)
                            preds += torch.nn.functional.pad(crop_upsampled_logits,
                                        (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                            count_mat[:, :, y1:y2, x1:x2] += 1
                    assert (count_mat == 0).sum() == 0
                    preds = preds / count_mat

                    # loss, upsampled_logits = model(pixel_values=images, bin_masks=binmasks, classes=classes, return_logits=True)

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
                    try:
                        tb_writer.add_scalar(f"Loss ({val_name}_val): ", mloss, i_iter)
                        tb_writer.add_scalar(f"mIoU ({val_name}_val): ", miou, i_iter)
                        tb_writer.add_scalar(f"mAcc ({val_name}_val): ", macc, i_iter)
                    except:
                        pass

                del upsampled_logits, labels
