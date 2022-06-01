'''
What this program do:
- Vocabulary mapping from each word to index
- Setup pytorch dataloader
- Setup padding for every batch
'''

# Library Import
import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_treshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        
        # get img and captionn
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Intialize Vocabulary
        self.vocab = Vocabulary(freq_treshold)
        self.vocab.build_vocabulary(self.captions.tolist())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.caption[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)