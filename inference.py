from mmseg.apis import MMSegInferencer
from glob import glob
import cv2
import os
from tqdm import tqdm
import argparse


def split_indices(indices):
    num_parts = 4
    total_indices = len(indices)
    part_size = total_indices // num_parts
    
    divided_indices = []
    start = 0
    end = part_size

    for _ in range(num_parts - 1):
        divided_indices.append(indices[start:end])
        start = end
        end += part_size

    # Add the remaining indices to the last part
    divided_indices.append(indices[start:])

    return divided_indices


def main():
    parser = argparse.ArgumentParser(description='Divide list of indices into roughly four parts and retrieve a specific section.')
    parser.add_argument('section', type=int, help='Section number (0-3) to retrieve')
    args = parser.parse_args()
    section_number = args.section

    print("Start Inference")
    config_path = '/home/mykang/mmsegmentation/_satellite/deeplabv3plus_r101_ver1/deeplabv3plus_ade20k_r101_d8.py'
    checkpoint_path = '/home/mykang/mmsegmentation/_satellite/deeplabv3plus_r101_ver1/iter_25000.pth'


    # Load models into memory
    mmseg_inferencer = MMSegInferencer(
        config_path,
        checkpoint_path,
        dataset_name="satellite",
        device='cuda')


    # Inference
    test_images = glob('/home/mykang/mmsegmentation/data/Satellite/img_dir/test/*.png')

    divided_indices = split_indices(test_images)
    test_images_section = divided_indices[section_number]

    for img_path in tqdm(test_images_section):
        result = mmseg_inferencer(img_path)
        
        cv2.imwrite(f"/home/mykang/mmsegmentation/inference_/20230706_deeplabv3plus_r101_ver1/{os.path.basename(img_path)}", result['predictions'])


if __name__ == "__main__":
    main()