import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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


# Function to remove segments with size 1 (4-connectivity)
def remove_small_segments(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    # Iterate through the connected components
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area == 1:
            mask[labels == label] = 0
    return mask



submission = pd.read_csv("result.csv")

for idx in tqdm(range(submission.shape[0])):
    test_id, mask_rle = submission.iloc[idx]


    gt_mask = rle_decode(mask_rle, (224, 224))

    # Remove segments with size 1
    modified_mask = remove_small_segments(gt_mask)

    # Perform RLE encoding again
    new_mask_rle = rle_encode(modified_mask)

    # Update the RLE mask in the submission dataframe
    submission.at[idx, 'mask_rle'] = new_mask_rle


submission.to_csv("result_1pix.csv", index=False)