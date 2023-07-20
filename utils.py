import numpy as np
import pandas as pd
import torch
from typing import List, Union
from joblib import Parallel, delayed

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def dice_coef(preds, true, eps=1e-7):
    preds = torch.sigmoid(preds)
    true = true.type_as(preds)
    intersection = (preds * true).sum()
    dsc = (2. * intersection) / (preds.sum() + true.sum() + eps)
    return dsc


# 제출 함수
def submission(column, result):
    submit = pd.read_csv("./data/sample_submission.csv")
    submit["mask_rle"] = result
    submit.to_csv("./submit.csv", index=False)
