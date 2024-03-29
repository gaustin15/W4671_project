from __future__ import absolute_import, division, unicode_literals
 
import sys
import io
import numpy as np
import logging
import argparse
import torch
import random
 
from transformers import *
import re


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
 
class generate_embedding():
 
    def __init__(self, embed_method, masks):
        # Select from embedding methods
        switcher = {
            'ave_last_hidden': self.ave_last_hidden,
            'CLS': self.CLS,
            'dissecting': self.dissecting,
            'ave_one_layer': self.ave_one_layer,
        }
       
        self.masks = masks
        self.embed = switcher.get(embed_method, 'Not a valide method index.')
 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 'ave_last_hidden': self.ave_last_hidden,
    def ave_last_hidden(self, params, all_layer_embedding):
        """
            Average the output from last layer
        """
        unmask_num = np.sum(self.masks, axis=1) - 1 # Not considering the last item
       
        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][-1,:,:]
            embedding.append(np.mean(hidden_state_sen[:sent_len,:], axis=0))
 
        embedding = np.array(embedding)
        return embedding
 
 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 'ave_last_hidden': self.ave_last_hidden,
    def ave_one_layer(self, params, all_layer_embedding):
        """
            Average the output from last layer
        """
        unmask_num = np.sum(self.masks, axis=1) - 1 # Not considering the last item
       
        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][params['layer_start'],:,:]
            embedding.append(np.mean(hidden_state_sen[:sent_len,:], axis=0))
 
        embedding = np.array(embedding)
        return embedding
 
 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 'CLS': self.CLS,
    def CLS(self, params, all_layer_embedding):
        """
            CLS vector as embedding
        """
        unmask_num = np.sum(self.masks, axis=1) - 1 # Not considering the last item
       
        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][-1,:,:]
            embedding.append(hidden_state_sen[0])
 
        embedding = np.array(embedding)
        return embedding
 
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 'dissecting': self.dissecting,
    def dissecting(self, params, all_layer_embedding):
        """
            dissecting deep contextualized model
        """
        unmask_num = np.sum(self.masks, axis=1) - 1 # Not considering the last item
        all_layer_embedding = np.array(all_layer_embedding)[:,params['layer_start']:,:,:] # Start from 4th layers output
 
        embedding = []
        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index,:,:unmask_num[sent_index],:]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:,token_index,:]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(params, token_feature)
                one_sentence_embedding.append(token_embedding)
 
            one_sentence_embedding = np.array(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(params, sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)
 
        embedding = np.array(embedding)
 
        return embedding
 
    def unify_token(self, params, token_feature):
        """
            Unify Token Representation
        """
        window_size = params['context_window_size']
 
        alpha_alignment = np.zeros(token_feature.shape[0])
        alpha_novelty = np.zeros(token_feature.shape[0])
       
        for k in range(token_feature.shape[0]):
        
            left_window = token_feature[k-window_size:k,:]
            right_window = token_feature[k+1:k+window_size+1,:]
            window_matrix = np.vstack([left_window, right_window, token_feature[k,:][None,:]])
           
            Q, R = np.linalg.qr(window_matrix.T) # This gives negative weights
 
            q = Q[:, -1]
            r = R[:, -1]
            alpha_alignment[k] = np.mean(normalize(R[:-1,:-1],axis=0),axis=1).dot(R[:-1,-1]) / (np.linalg.norm(r[:-1]))
            alpha_alignment[k] = 1/(alpha_alignment[k]*window_matrix.shape[0]*2)
            alpha_novelty[k] = abs(r[-1]) / (np.linalg.norm(r))
           
        
        # Sum Norm
        alpha_alignment = alpha_alignment / np.sum(alpha_alignment) # Normalization Choice
        alpha_novelty = alpha_novelty / np.sum(alpha_novelty)
 
        alpha = alpha_novelty + alpha_alignment
       
        alpha = alpha / np.sum(alpha) # Normalize
       
        out_embedding = token_feature.T.dot(alpha)
        
 
        return out_embedding
 
    def unify_sentence(self, params, sentence_feature, one_sentence_embedding):
        """
            Unify Sentence By Token Importance
        """
        sent_len = one_sentence_embedding.shape[0]
 
        var_token = np.zeros(sent_len)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:,token_index,:]
            sim_map = cosine_similarity(token_feature)
            var_token[token_index] = np.var(sim_map.diagonal(-1))
 
        var_token = var_token / np.sum(var_token)
 
        sentence_embedding = one_sentence_embedding.T.dot(var_token)
 
        return sentence_embedding
        
        
        
        
        
        
def create_embeddings(sentences, model, tokenizer, use_sbert = True):
    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences if type(s) == str]
    features_input_ids = []
    features_mask = []
    max_seq_len = 50
    for sent_ids in sentences_index:
            # Truncate if too long
        if len(sent_ids) > max_seq_len:
            sent_ids = sent_ids[: max_seq_len]
        sent_mask = [1] * len(sent_ids)
            # Padding
        padding_length = max_seq_len - len(sent_ids)
        sent_ids += [0] * padding_length
        sent_mask += [0] * padding_length
            # Length Check
        assert len(sent_ids) == max_seq_len
        assert len(sent_mask) == max_seq_len
            
        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)


    features_mask = np.array(features_mask)
    device = "cpu"
    batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
    batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
    batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}        
        
    if use_sbert:    
        model.zero_grad()

        with torch.no_grad():
            features = model(**inputs)[2]


        all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

        params = {'layer_start':4,'context_window_size':4,'embed_method':'dissecting'}

        embed_method = generate_embedding(params['embed_method'], features_mask)
        embedding = embed_method.embed(params, all_layer_embedding)
        
    else:
        embedding = model(**inputs)[1]
    
    return(embedding)