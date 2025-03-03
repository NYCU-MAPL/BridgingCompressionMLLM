from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
import os
from torch.utils.data.dataset import Dataset
import json



class coco_Karpathy(Dataset):
  def __init__(self, 
               labels_paths = '/path/to/Karpathy_split.json', 
               base_image_dir = '/path/to/coco/',
               split: str = None,
               transform = None):

    self.base_image_dir = base_image_dir
    print(f"Load data from {self.base_image_dir}")

    self.images = []
    self.captions = []
    self.cocoids = []
    f = open(labels_paths)
    data = json.load(f)
    for img in data['images']:
        if img['split'] == split:
            self.images.append(os.path.join(img['filepath'], img['filename']))
            self.captions.append(img['sentences'][random.randint(0,4)]['raw'])
            self.cocoids.append(img['cocoid'])
        elif split == 'train' and img['split'] == 'restval':
            self.images.append(os.path.join(img['filepath'], img['filename']))
            self.captions.append(img['sentences'][0]['raw'])
            self.cocoids.append(img['cocoid'])
    
    self.transform = transform
        
    assert len(self.images) == len(self.captions)
    assert len(self.images) == len(self.cocoids)
    assert len(self.cocoids) == len(self.captions)

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):

    image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
    img = Image.open(image_path).convert('RGB')
    images = self.transform(img)

    caption = str(self.captions[idx])

    cocoid = self.cocoids[idx]

    return images, caption, cocoid