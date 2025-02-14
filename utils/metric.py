import torch

from utils.colors import CITY_VALID_CLASSES


def get_confBins(predictions, references, ignore_index, n_classes=19):
    predictions = predictions.flatten()
    references = references.flatten()

    predictions = predictions[references != ignore_index]
    references = references[references != ignore_index]

    bin_p = torch.bincount(predictions, minlength=n_classes)
    bin_t = torch.bincount(references, minlength=n_classes)

    hits = predictions[predictions == references]
    bin_hits = torch.bincount(hits, minlength=n_classes)

    confBins = torch.stack((bin_hits, bin_p, bin_t))
    return confBins

def get_metrics(confBins, dice_smooth=1):
    bin_hits, bin_p, bin_t = confBins

    union = bin_p + bin_t - bin_hits

    jaccard = bin_hits / union
    accuracy = bin_hits / bin_t

    jaccard[bin_t == 0] = torch.nan

    # dice = (2.* bin_hits.sum() + dice_smooth) / (bin_p.sum() + bin_t.sum() + dice_smooth)

    return jaccard, accuracy#, 1-dice

def perclass_repr(res):
    print("\nPer-class results (Val):")
    for i, (iou, acc) in enumerate(res):
        print('Class {} {} IoU {:.2f} Acc {:.2f}'.format(i, CITY_VALID_CLASSES[i], iou, acc))