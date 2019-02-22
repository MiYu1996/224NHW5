#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embeddings_char = nn.Embedding(len(vocab.char2id),50,padding_idx = vocab.char2id['<pad>'])
        self.CNN = CNN(50,21,embed_size)
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(p=0.3)
        self.embed_size = embed_size
        
        

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        ####when create embedding, we can combine all the sentence_length
        ####We will loop through the entire sentence using split
        x_padded = input
        x_split = torch.split(x_padded,1,dim = 0)

        x_word_emb = []
        for x in x_split:
            ####This will get rid of the first dimension
            x = torch.squeeze(x,dim = 0)
            x_emb = self.embeddings_char(x)
            x_reshape = x_emb.permute(0,2,1)
            x_conv_out = self.CNN(x_reshape)
            ####And we will need to get rid of the last dimension in order to input
            ####into our highway layer
            x_conv_out = torch.squeeze(x_conv_out,dim = -1)
            x_highway = self.highway(x_conv_out)
            x_result = self.dropout(x_highway)
            x_word_emb.append(x_result)
        x_word_emb = torch.stack(x_word_emb,dim = 0)
        
        return x_word_emb


        
        

        ### END YOUR CODE

