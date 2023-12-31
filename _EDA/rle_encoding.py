import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()

    pixels = pixels - 7
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


submission = pd.read_csv("/home/mykang/mmsegmentation/data/sample_submission.csv")

for idx in tqdm(range(submission.shape[0])):
    img_id = submission['img_id'].iloc[idx]

    mask = cv2.imread(f"/home/mykang/mmsegmentation/_satellite/conv_ver6/format_results/{img_id}.png", cv2.IMREAD_GRAYSCALE)

    rle_encoding = rle_encode(mask)

    if rle_encoding == '':
        rle_encoding = -1
    
    submission.at[idx, 'mask_rle'] = rle_encoding


submission.to_csv("result.csv", index=False)