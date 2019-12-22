#!/usr/bin/env python
# coding: utf-8



from models import *
from solver import *


# Data parameters
data_folder = './inputData/'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
    
benchmark = False # if GPU is rtx2080, else True


#NIC Model parameters
embed_encoder_dim = 512  # dimension of word embeddings
hidden_dim = 512  # dimension of decoder RNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

#NICA Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
hidden_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# encoder_model = 'vgg16', 'densenet161',  'resnet101'

encoder_model = 'resnet101'
encoder = Encoder(encoder_model = encoder_model)

# decoder_model = 'NIC', 'NICA'

decoder_model= 'NICA'


if decoder_model == 'NICA':
    decoder = DecoderWithAttention(attention_dim=attention_dim, embed_dim=emb_dim, hidden_dim=hidden_dim,
                                       vocab_size=len(word_map), encoder_dim = encoder.output_dim, 
                                       dropout=dropout, device = device)
elif decoder_model == 'NIC':
    decoder = DecoderWithoutAttention(image_output_dim = encoder.output_dim,hidden_dim = hidden_dim, vocab_size =len(word_map),
                                          embed_encoder_dim=embed_encoder_dim, device = device)

checkpoint = 'BEST_checkpoint_resnet101_NICA_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# checkpoint = None


backprop_deep(encoder, decoder, data_folder, data_name, word_map, 
              epochs = 120, decoder_lr = 4e-4 , checkpoint = checkpoint, device = device, benchmark = benchmark)


