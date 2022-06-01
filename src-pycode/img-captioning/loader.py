'''
What this program do:
- Vocabulary mapping from each word to index
- Setup pytorch dataloader
- Setup padding for every batch
'''

# Library Import
import os
import json
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

spacy_eng = spacy.load('en_core_web_sm')

def downloaddata():
    json_file_kaggle = json.load(open("./data/flickr8k/kaggle.json"))
    os.environ['KAGGLE_USERNAME'] = json_file_kaggle['username']
    os.environ['KAGGLE_KEY'] = json_file_kaggle['key']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('adityajn105/flickr8k', path="./data/flickr8k")
    print("Download Complete")

    from zipfile import ZipFile
    file_name = "./data/flickr8k/flickr8k.zip"
    with ZipFile(file_name, 'r') as zip:
        print('Extracting all the files now...')
        zip.extractall(path="./data/flickr8k")
        print('Done!')

class Vocabulary:
    def __init__(self, freq_treshold):
        self.itos = {0:'<PAD', 1:'<SOS', 2:'<EOS>', 3:'<UNK>'}
        self.stoi = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.freq_treshold = freq_treshold
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        """
            * The function takes in a string of text, then performs the following:
            * Loads the `spacy` model
            * Tokenizes the text and lemmatizes it
            * Removes stopwords
            * Returns a list of the cleaned text
        
        Let's give it a try:
        
        :param text: The text we want to tokenize
        :return: A list of tokens in lowercase.
        
        Example
            Input: "I love my dog"
            Return: ['i', 'love', 'my', 'dog']
        """
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        """
        > For each word in each sentence in the list of sentences, if the word is not in the dictionary, add
        it to the dictionary with a value of 1. If the word is already in the dictionary, increment its
        value by 1. If the word's value is equal to the frequency threshold, add it to the itos and stoi
        dictionaries
        
        :param sentence_list: list of sentences
        """
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                    
                else:
                    frequencies[word] += 1
                    
                if frequencies[word] == self.freq_treshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1
    
    def numericalize(self, text):
        """
        **The numericalize function takes in a string of text and returns a list of integers, where each
        integer corresponds to a token in the text.**
        
        Let's see how this works in practice
        
        :param text: The text to be tokenized
        :return: A list of integers, where each integer is the index of the token in the vocabulary.
        """
        tokenized_text = self.tokenizer_eng(text)
        
        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text]
    

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
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        
        return imgs, targets
    
def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):

    dataset = FlickerDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    
    return loader, dataset


def main():
    downloaddata()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    dataloader = get_loader("./data/flickr8k/Images", annotation_file="./data/flickr8k/captions.txt", transform=transform)
    for idx, (img, caption) in enumerate(dataloader):
        print(img.shape, caption.shape)
        
if __name__ == "__main__":
    main()