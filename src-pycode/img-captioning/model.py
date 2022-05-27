# -- import -- #

from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models

# -- class model -- #


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super().__init__()
        self.train_CNN = train_CNN

        """
        Models Source: https://pytorch.org/vision/stable/generated/torchvision.models.inception_v3.html
        """

        self.inception = models.inception_v3(pretrained=True, aux_logits=False)

        # replace the last layer of the inception model with nn.Linear
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images):
        features = self.inception(images)

        """
        Creating a conditional case for training the CNN.
        If the argument passed into the class is True, then the CNN is trained by activating the ```requires_grad``` flag.
        Default = False -> CNN is not trained, only the fc layer is trained.
        """
        for name, param in self.inception.named_parameters():
            if name == "fc.weight" or name == "fc.bias":
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        output = self.relu(features)
        output = self.dropout(output)
        return output


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

        hiddens, _ = self.lstm(embeddings)
        output = self.linear(hiddens)
        return output
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, caption):
        features = self.encoder(images)
        output = self.decoder(features, caption)
        return output
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        with torch.no_grad:
            x = self.encoder(image).unsqueeze(0)
            states = None
            
            for _ in range(max_length):
                hidden, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hidden.unsqueeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)
                
                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break
                
        return [vocabulary.itos[i] for i in result_caption]