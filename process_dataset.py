import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
import random
from utils.rgb_ind_convertor import floorplan_fuse_map, rgb2ind

# Configuration
IMAGE_SIZE = (512, 512)  # (height, width)

def process_dataset(dataset_type):
    # dataset_path = f"./dataset/r3d_{dataset_type}.txt"
    dataset_path = f"./dataset/r3d_{dataset_type}.txt"
    print("Processing dataset:", dataset_path)

    paths = open(dataset_path,"r").read().splitlines()
    image_paths = [p.split('\t')[0] for p in paths] # image 
    gt1 = [p.split('\t')[1] for p in paths] # 1 wall
    gt2 = [p.split('\t')[2] for p in paths] # 2 window,door
    gt3 = [p.split('\t')[3] for p in paths] # 3 rooms
    gt4 = [p.split('\t')[-1] for p in paths] # close wall

    for idx, image_path in enumerate(tqdm(image_paths)):
        img = cv2.imread(image_path[1:])
        wall = cv2.imread(gt1[idx][1:],0)
        door = cv2.imread(gt2[idx][1:],0)
        room = rgb2ind(cv2.imread(gt3[idx][1:])[:,:,::-1],
                color_map=floorplan_fuse_map)
        boundary = np.zeros(door.shape)
        boundary[door>0] = 1
        boundary[wall>0] = 2
        image = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST_EXACT).astype(np.uint8)
        boundary = cv2.resize(boundary, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST_EXACT).astype(np.uint8)
        room = cv2.resize(room, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST_EXACT).astype(np.uint8)
        door = cv2.resize(door, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST_EXACT).astype(np.uint8)

        # Skip invalid samples
        unique_rooms = np.unique(room).tolist()
        if any(room_id > 9 or room_id < 0 for room_id in unique_rooms):
            continue
        
        if not os.path.exists(f"np_dataset/{dataset_type}"):
            os.makedirs(f"np_dataset/{dataset_type}")
        sample_name = Path(image_path).name
        save_dict = {
            "image": image,
            "boundary": boundary,
            "room": room,
            "door": door,
        }
        np.savez(f"np_dataset/{dataset_type}/{sample_name}.npz", **save_dict)

def plot_sample(npz_path):
    """Load and plot a sample from the processed dataset"""
    data = np.load(npz_path)
    
    # Get data
    image = data['image']
    boundary = data['boundary']
    room = data['room']
    door = data['door']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot original image
    axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # Plot boundary (walls and doors)
    axes[0,1].imshow(boundary, cmap='tab20')
    axes[0,1].set_title('Boundary (0:background, 1:door, 2:wall)')
    axes[0,1].axis('off')
    
    # Plot room labels
    axes[1,0].imshow(room, cmap='tab20')
    axes[1,0].set_title('Room Labels')
    axes[1,0].axis('off')
    
    # Plot doors
    axes[1,1].imshow(door, cmap='gray')
    axes[1,1].set_title('Doors')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Process datasets if needed
    for dataset_type in ["train", "test"]:
        if not os.path.exists(f"np_dataset/{dataset_type}"):
            process_dataset(dataset_type)
    
    # Plot a sample from the test set
    test_samples = os.listdir("np_dataset/test")
    if test_samples:
        for i in range(10):
            idx = random.randint(0, len(test_samples))
            sample_path = os.path.join("np_dataset/test", test_samples[idx])
            print(f"Plotting sample: {sample_path}")
            plot_sample(sample_path)