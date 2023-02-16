from torch.utils.data import Dataset
import os
from PIL import Image

class Custom_Dataset(Dataset):

  def __init__(self, root, txt, transform=None, returnPath=False, pre_load=False, pathReplace={}):
    self.img_path = []
    self.labels = []
    self.root = root
    self.transform = transform
    self.returnPath = returnPath
    self.txt = txt

    with open(txt) as f:
      for line in f:
        label = line.split()[-1]
        self.img_path.append(os.path.join(root, line[:-(len(label) + 2)]))
        self.labels.append(int(label))

    for key, item in pathReplace.items():
      self.img_path = [p.replace(key, item) for p in self.img_path]

    self.pre_load = pre_load
    if pre_load:
      self.imgs = {}
      print("preloading images")
      for idx in range(len(self.img_path)):
        if idx % 100 == 0 and idx > 0:
          print("loading {}/{}".format(idx, len(self.img_path)))
        path = self.img_path[idx]
        with open(path, 'rb') as f:
          sample = Image.open(f).convert('RGB')
        self.imgs[idx] = sample

    self.targets = self.labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):

    path = self.img_path[index]
    label = self.labels[index]

    if not self.pre_load:
      with open(path, 'rb') as f:
        sample = Image.open(f).convert('RGB')
    else:
      sample = self.imgs[index]

    if self.transform is not None:
      sample = self.transform(sample)

    if not self.returnPath:
      return sample, label
    else:
      return sample, label, index, path.replace(self.root, '')


