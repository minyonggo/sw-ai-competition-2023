import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


df = pd.read_csv("/home/mykang/mmsegmentation/data/train_split.csv")

for index in tqdm(range(df.shape[0])):
    train_id, img_path, rle_mask, split = df.iloc[index]

    img = cv2.imread(f"/home/mykang/mmsegmentation/data/train_img/{train_id}.png")
    gt_mask = rle_decode(rle_mask, img.shape[:2])

    if split == 0:
        cv2.imwrite(f"/home/mykang/mmsegmentation/data/Satellite/img_dir/train/{train_id}.png", img)
        cv2.imwrite(f"/home/mykang/mmsegmentation/data/Satellite/ann_dir/train/{train_id}.png", gt_mask)

    else:
        cv2.imwrite(f"/home/mykang/mmsegmentation/data/Satellite/img_dir/val/{train_id}.png", img)
        cv2.imwrite(f"/home/mykang/mmsegmentation/data/Satellite/ann_dir/val/{train_id}.png", gt_mask)

