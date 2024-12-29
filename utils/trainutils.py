import random
import numpy
import torch

def fix_seed(SEED, deterministic=True):
    random.seed(SEED)
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    op = torch.stack if batch[0]["img"].dim() == 3 else torch.cat
    output = {
        "image": op([x["img"] for x in batch]),
        "label": op([x["lbl"] for x in batch]),
        "fname": [x["fname"] for x in batch]
    }

    if "classes" in batch[0].keys():
        if isinstance(batch[0]["classes"], list):
            output.update({
                "classes": [y for x in batch for y in x["classes"]], 
                "bin_masks": [y for x in batch for y in x["bin_masks"]]
            })
        else:
            output.update({
                "classes": [x["classes"] for x in batch], 
                "bin_masks": [x["bin_masks"] for x in batch]
            })
        
    return output

def get_batch(iter_, loader):
    try:
        batch = next(iter_)
    except:
        iter_ = iter(loader)
        batch = next(iter_)
    return batch, iter_

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(lr, lr_power, i_iter, warmup_iters, max_iterations, optimizer):
    if i_iter < warmup_iters:
        learning_rate = float(i_iter / warmup_iters * lr)
    else:
        learning_rate = lr_poly(lr, i_iter-warmup_iters, max_iterations-warmup_iters, lr_power)

    for param in optimizer.param_groups:
        if param["name"] in ["vision_decoder", "neck"]:
            param['lr'] = lr * 10
        else:
            param['lr'] = learning_rate
