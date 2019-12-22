#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision as tv
from torch import nn



class Encoder(nn.Module):
    '''
    Use CNN model to extract the feature of image
    '''
    def __init__(self, encoder_model = 'vgg16', encoded_image_size=14, fine_tuning=False):
        super(Encoder, self).__init__()
        self.encoder_model = encoder_model
        self.encoded_image_size = encoded_image_size
        if encoder_model == 'vgg16':
            vgg = tv.models.vgg16_bn(pretrained=True)   
            self.features = vgg.features[:-1]
            self.output_dim = 512
        elif encoder_model == 'densenet161':
            densenet = tv.models.densenet161(pretrained=True)
            modules = list(densenet.children())[:-1]
            self.features = nn.Sequential(*modules)
            self.output_dim = 2208
        elif encoder_model == 'resnet101':
            resnet = tv.models.resnet101(pretrained=True)
            modules = list(resnet.children())[:-2]
            self.features = nn.Sequential(*modules)
            self.output_dim = 2048
        for param in self.features.parameters():
            param.requires_grad = fine_tuning 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
         
            
        
    def forward(self, x): 
        y = self.features(x) # (batch_size, 512, encoded_image_size, encoded_image_size)
        y = self.adaptive_pool(y) # (batch_size, 512, encoded_image_size, encoded_image_size)
        y = y.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 512)
        return y
    
    
    
class DecoderWithoutAttention(nn.Module):
    '''
    Use LSTM model as decoder
    '''

    def __init__(self, image_output_dim, hidden_dim, vocab_size, 
                 embed_encoder_dim=512, dropout=0.5, encoded_image_size=14, device = 'cuda'):
        """
        :param embed_encoder_dim: embedding size and feature size of encoded images
        :param hidden_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param dropout: dropout
        """
        super(DecoderWithoutAttention, self).__init__()
        self.decoder_model = 'NIC'
        self.embed_encoder_dim = embed_encoder_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device


        self.embedding = nn.Embedding(vocab_size, embed_encoder_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_encoder_dim, hidden_dim, bias=True)  # decoding LSTMCell
        self.fc = nn.Linear(hidden_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.image_encoding = nn.Linear(image_output_dim * encoded_image_size**2, embed_encoder_dim)  # linear layer to find scores over vocabulary
    
        # Initializes parameters
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        
        batch_size = encoder_out.size(0)
        embed_encoder_dim = self.embed_encoder_dim
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.contiguous().view(batch_size, -1)  # (batch_size, num_pixels*embed_encoder_dim)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        encoder_out = self.image_encoding(encoder_out)  # (batch_size, embed_encoder_dim)
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_encoder_dim)
        
        # Initialize LSTM state
        h = torch.zeros(batch_size, self.hidden_dim).to(self.device)  # (batch_size, hidden_dim)
        c = torch.zeros(batch_size, self.hidden_dim).to(self.device)  # (batch_size, hidden_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)

        # At each time-step, decode by
        # the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t-1 for l in decode_lengths])
            
            if t == 0:
                h, c = self.decode_step(encoder_out[:batch_size_t, :], 
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_dim)
            else:
                h, c = self.decode_step(embeddings[:batch_size_t, t-1, :],
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind
    
    

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param hidden_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, hidden_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """    
    def __init__(self, attention_dim, embed_dim, hidden_dim, vocab_size, encoder_dim, dropout=0.5, device = 'cuda'):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param hidden_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        self.decoder_model = 'NICA'
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device

        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, hidden_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, hidden_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(hidden_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_dim, vocab_size)  # linear layer to find scores over vocabulary



        #Initializes some parameters with values from the uniform distribution, for easier convergence.
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)



    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, hidden_dim)
        c = self.init_c(mean_encoder_out)        

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    