from importmod import *
import pandas as pd
import random
import os
from pathlib import Path
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader

class MyRotationTransform:
    def __init__(self,angles=[0,90,-90.180]):
        self.angles = angles
    def _r(self,x,angle):
        return rotate(x,angle,preserve_range=True)
    def __call__(self,x,y,z,g):
        angle = random.choice(self.angles)
        return self._r(x,angle),self._r(y,angle),self._r(z,angle),self._r(g,angle)

class r3dDataset(Dataset):
    def __init__(self,size=512,transform=None):
        self.size = size
        self.transform = transform
        self.rotation = MyRotationTransform()

        self.train_dir = Path("np_dataset/train")
        self.test_dir = Path("np_dataset/test")
        self.train_paths = list(self.train_dir.iterdir())
        self.test_paths = list(self.test_dir.iterdir())

    def __len__(self):
        return len(self.train_paths) + len(self.test_paths)
    
    def _getset(self,idx):
        target = self.train_paths if idx < len(self.train_paths) else self.test_paths
        idx = idx if idx < len(self.train_paths) else idx - len(self.train_paths)

        sample = np.load(target[idx])

        return sample["image"], sample["boundary"], sample["room"], sample["door"]
    
    def __getitem__(self,idx):
        image, boundary, room, door = self._getset(idx)
        if self.transform:
            try:
                image = self.transform(image.astype(np.float32)/255.0)
                boundary = self.transform(F.one_hot(
                    torch.LongTensor(boundary),3).numpy())
                room = self.transform(F.one_hot(
                    torch.LongTensor(room),9).numpy())
                door = self.transform(door)
            except RuntimeError:
                print(np.unique(boundary))
                print(np.unique(room))
                raise

        return image, boundary, room, door

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    DFPdataset = r3dDataset()
    image,boundary,room,door = DFPdataset[200]

    plt.subplot(2,2,1); plt.imshow(image)
    plt.subplot(2,2,2); plt.imshow(boundary)
    plt.subplot(2,2,3); plt.imshow(room)
    plt.subplot(2,2,4); plt.imshow(door)
    plt.show()

    breakpoint()
    
    gc.collect()

