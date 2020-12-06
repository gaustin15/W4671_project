#this file contains encoder structures/models.
#all teh structures are written by Irfan,
# with the full model structures (i.e. those that take "batch" as an input)
#       beign written by me (George)
 
import numpy as np
import pandas as pd
import pickle
 
import torch
import torch
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
import torchtext.data as textdata
from torchtext.data import TabularDataset
from torchtext.data import Field,NestedField,LabelField
from torchtext.data import Iterator, BucketIterator
 
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
 
import gc
import matplotlib.pyplot as plt
 
from collections import OrderedDict
 
import math
import torch.nn.functional as F
 
 
 
class StructuredSelfAttention(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_hops,dropout=0.1):
        super().__init__()
        self.linear_1=nn.Linear(model_dim,hidden_dim,bias=False)
        self.linear_2=nn.Linear(hidden_dim,num_hops,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.init_weights()
        self.num_hops=num_hops
        self.linear_out=nn.Linear(num_hops*model_dim,model_dim,bias=False)
        eye=torch.eye(self.num_hops)
        self.register_buffer('eye',eye) # ensure correct device behaviour
        self.softmax=nn.Softmax(dim=-2)
    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
 
    def forward(self, x, mask=None):
        """
        mask (LongTensor): [batch_size x src_len x 1 x feat_len]
        input (FloatTensor): [batch_size x src_len x feat_len x d_model]
        lenghs (FloatTensor): [batch_size x src_len]
        """
        input_rep = x # [batch_size x src_len x feat_len x d_model]
        batch_size,src_len,feat_len,model_dim=x.size()
        x = self.dropout(x)
        x__ = self.linear_1(x) # [batch_size x src_len x feat_len x hidden_dim]
        x = torch.tanh(x__)
        x = self.linear_2(x) # [batch_size x src_len x feat_len x attn_hops]
        if mask is not None:
            mask=mask.squeeze(2).unsqueeze(3) # [batch_size x src_len x feat_len x 1]
            x = x.masked_fill(mask, -1e18)
 
        x = self.softmax(x) # [batch_size x src_len x feat_len x attn_hops]
#         claim_lengths=mask.squeeze(3).data.ne(1).sum(dim=2) #  [batch_size x src_len]
#         # the scores for 0 code length visits will be 1/feat_len
#         # mask fill all the scores for the 0-length visits
#         visits_mask = claim_lengths.data.eq(0).unsqueeze(2).unsqueeze(3) #  [batch_size x src_len x 1 x 1]
#         x=x.masked_fill(visits_mask,0) # [batch_size x src_len x feat_len x attn_hops]
       
        visit_rep=torch.matmul(x.transpose(2,3).contiguous(),x__)#input_rep)
        #visit_rep=visit_rep.view(batch_size,src_len,self.num_hops*model_dim)
        #visit_rep=self.linear_out(visit_rep) # [batch_size x src_len x model_dim]
 
        if self.num_hops==1:
            penalty=None
        else:
            penalty=torch.norm(torch.matmul(x.transpose(-2,-1),x)-self.eye)
        return visit_rep, penalty
   
    def update_dropout(self,dropout):
        self.dropout.p=dropout
 
class PositionalEncoding(nn.Module):
    """
    dim (int): embedding size
    dropout (float): dropout rate
    """
    def __init__(self,dropout,dim,max_len=20*365,pos_emb_type="learned",padding_idx=None,sparse=False):
        super().__init__()
        if pos_emb_type=="sine":
            pe=torch.zeros(max_len,dim) #[max_len x dim]
            position=torch.arange(0,max_len).unsqueeze(1) #[max_len x 1]
            div_term=torch.exp(torch.arange(0,dim,2,dtype=torch.float)*
                          -(math.log(10000.0)/dim)) # [dim/2]
            pe[:,0::2]=torch.sin(position.float()*div_term) #[max_len x dim/2]
            pe[:,1::2]=torch.cos(position.float()*div_term) #[max_len x dim/2]
            if padding_idx is not None:
                pe[padding_idx, :] = 0
            pe=pe.unsqueeze(0)
            self.register_buffer('pe',pe)
            self.dim=dim
        else:
            self.embedding=nn.Embedding(500,dim,sparse=sparse)
        self.pos_emb_type=pos_emb_type
       
    def forward(self,emb,dates=None,step=None):
        '''
        emb (FloatTensor): [batch x src_len x model_dim]
        '''
        batch_size = emb.size(0)
        seq_len = emb.size(1)
 
        if dates is not None:
            if self.pos_emb_type=="sine":
                emb=emb+torch.index_select(self.pe,1,dates.view(-1)).view(batch_size,seq_len,-1)
            else:
                emb+=self.embedding(dates.long())
        else:
            emb=emb+self.pe[:,:emb.size(0),:]
 
        return emb
 
class Embeddings(nn.Module):
    """
    word_vec_size (int): dimensionality of embeddings
    word_padding_idx (int): word index for padding
    word_vocab_size (int): size of the vocabulary
    position_encoding (bool):
    dropout (float): dropout rate
    """
    def __init__(self,word_vec_size,word_vocab_size,
                 word_padding_idx,
                position_encoding=False,
                 fix_word_vecs=False,
                dropout=0.1,
                sparse=False):
        super().__init__()
        self.word_padding_idx=word_padding_idx
        self.word_vec_size=word_vec_size
        embeddings=nn.Embedding(word_vocab_size,
                               word_vec_size,
                                padding_idx=word_padding_idx,
                               sparse=sparse)
        self.make_embeddings=nn.ModuleDict({'embedding_layer':embeddings})
        self.position_encoding=position_encoding
        if self.position_encoding:
            pe=PositionalEncoding(dropout,word_vec_size)
            self.make_embeddings.update({'positional_encoding':pe})
        if fix_word_vecs:
            self.make_embeddings.embedding_layer.weight.requires_grad=False
       
    def load_pretrained_vectors(self,emb_file,emb_factor):
        """
        emb_tensor (FloatTensor) : pretrained word vectors [vocab_size x vect_size]
        """
        if emb_file is not None:
            emb_tensor=torch.load(emb_file)
            pretrained_vec_size=emb_tensor.size(1)
            if emb_factor is not None and emb_factor!=1:
                m = nn.AvgPool1d(emb_factor,emb_factor)
                emb_tensor=m(emb_tensor.unsqueeze(0))
                emb_tensor=emb_tensor.squeeze(0).data
            if self.word_vec_size>pretrained_vec_size:
                self.make_embeddings.embedding_layer.weight.data[:,:pretrained_vec_size]=emb_tensor
            elif self.word_vec_size < pretrained_vec_size:
                self.make_embeddings.embedding_layer.weight.data.copy_(emb_tensor[:,:self.word_vec_size])
            else:
                self.make_embeddings.embedding_layer.weight.data.copy_(emb_tensor)
 
    def forward(self,source,dates=None,step=None):
        """
        source (LongTensor) : index tensor [len x batch x nfeat]
        out (FloatTensor) : word embeddings [len x batch x embedding_size]
        """
        if self.position_encoding:
            for key,module in self.make_embeddings.items():
                if key=='positional_encoding':
                    source=module(source,dates,step)
                else:
                    source=module(source)
                    source=source*math.sqrt(self.word_vec_size)
        else:
            source=self.make_embeddings.embedding_layer(source)
            source=source*math.sqrt(self.word_vec_size)
        return source
 
class StructuredAttentVisEncoder(nn.Module):
    def __init__(self,model_dim, hidden_dim, num_hops, embeddings,dropout=0.1):
        super().__init__()
        self.embeddings=embeddings
        self.selfattn=StructuredSelfAttention(model_dim, hidden_dim, num_hops,dropout=dropout)
        self.layer_norm=nn.LayerNorm(model_dim,eps=1e-6)
        self.dropout=nn.Dropout(dropout)
        self.elu=nn.ELU()
       
    def forward(self,src):
        emb=self.embeddings(src) # [batch_size x seq_len x feat_len x d_model
        emb=self.dropout(self.layer_norm(emb))
        w_batch,w_len,_= src.size()
        padding_idx=self.embeddings.word_padding_idx
        mask=src.data.eq(padding_idx).unsqueeze(2) #[batch_size x src_len x 1 x feat_len]
        lengths=torch.sum(src.data.ne(padding_idx),dim=-1) #[batch_size x src_len]
        out,penalty=self.selfattn(emb,mask)
       
        #output nonlinearity
        lengths_filled = lengths.masked_fill(lengths.data.eq(0),1)
        emb_cbow=torch.sum(emb,dim=2)
        emb_cbow_normed=torch.div(emb_cbow,torch.sqrt(lengths_filled.unsqueeze(2).float()))
        visit_rep=self.elu(self.dropout(out)+emb_cbow_normed)
       
        return visit_rep, lengths, penalty
 
class Demographics(nn.Module):
    def __init__(self,max_age,age_embedding_dim,sparse=False):
        super().__init__()
        self.age_embedding = nn.Embedding(max_age,age_embedding_dim,sparse=sparse)
        self.gnd_embedding = nn.Embedding(4,age_embedding_dim,sparse=sparse)
        self.layer_norm=nn.LayerNorm(2*age_embedding_dim,eps=1e-6)
    def forward(self,age,gnd):
        age_emb = self.age_embedding(age)
        gnd_emb = self.gnd_embedding(gnd)
        out = torch.cat((age_emb,gnd_emb),dim=-1)
        out = self.layer_norm(out)
        return out
 
class CompleteNetwork(nn.Module):
    def __init__(self, emb_dim = 128, hidden_size = 32, dem_emb_size = 2, num_hops = 1 ):
        super(CompleteNetwork, self).__init__()
 
        # try emb_dim = 64
        # hidden at 16 might be small
       
        # keep increasing sizes
 
        #embedding layer for proc tensors
        self.EMB_proc = Embeddings(word_vec_size = emb_dim,
                     word_vocab_size = len( proc_field.vocab ),
                     word_padding_idx = 0)
        #stuctured attention encoder
        self.SSA_proc = StructuredAttentVisEncoder(emb_dim, emb_dim, num_hops, self.EMB_proc )
       
        #throw in linear here
        ### combined both SSAs using linear ==> then throw in positional encoding on combination
       
        #positional encoder
        self.PE_proc = PositionalEncoding(dropout = 0.1, dim  = emb_dim, pos_emb_type = 'sine')
 
 
        #setup below is the same but for icd codes
        self.EMB_diag = Embeddings(word_vec_size = emb_dim,
                     word_vocab_size = len( visit_field.vocab ),
                     word_padding_idx = 0)
        self.SSA_diag = StructuredAttentVisEncoder(emb_dim, emb_dim, num_hops, self.EMB_diag )
        self.PE_diag = PositionalEncoding(dropout = 0.1, dim = emb_dim, pos_emb_type = 'sine')
 
        self.LSTM = nn.LSTM(input_size = emb_dim * 2, hidden_size = hidden_size, batch_first = True )
       
        #embedding the demographic info
        self.DEM = Demographics(150, dem_emb_size)
       
        self.linear =  nn.Linear(in_features = hidden_size + dem_emb_size * 2, out_features = 2)
 
    def forward(self, batch):
        prs = self.PE_proc(self.SSA_proc(batch.proc_cd)[0], dates = batch.dates )
        dis = self.PE_diag(self.SSA_diag(batch.diag_cd)[0], dates = batch.dates )
       
        #stacking the encoded procs/icd codes together
        # ===> I'm assuming this is ok since the indexes are now encoded,
        #      and doing this allows icd info to impact processing of future proc info...
       
        stacked = torch.cat((prs, dis), axis = 2)
       
        Lout = self.LSTM(stacked)[0][:, -1, :]
       
        #stacking on the demographics
        add_dems = torch.cat( (Lout, self.DEM(batch.age, batch.gender)), axis = 1)
        
        #add a linear layer to get to output dimensions
        out = F.softmax( self.linear(add_dems), dim = 1 )
        return(out)
       
 
 
 
class LinearPDCombo(nn.Module):
    def __init__(self, emb_dim = 64, hidden_size = 32, dem_emb_size = 2, num_hops = 2 ):
        super(LinearPDCombo, self).__init__()
 
        # try emb_dim = 64
        # hidden at 16 might be small
       
        # keep increasing sizes
 
        #embedding layer for proc tensors
        self.EMB_proc = Embeddings(word_vec_size = emb_dim,
                     word_vocab_size = len( proc_field.vocab ),
                     word_padding_idx = 0)
        #stuctured attention encoder
        self.SSA_proc = StructuredAttentVisEncoder(emb_dim, emb_dim, num_hops, self.EMB_proc )  
        
        self.EMB_diag = Embeddings(word_vec_size = emb_dim,
                     word_vocab_size = len( visit_field.vocab ),
                     word_padding_idx = 0)
        self.SSA_diag = StructuredAttentVisEncoder(emb_dim, emb_dim, num_hops, self.EMB_diag )
       
        self.L1 = nn.Linear(in_features = emb_dim, out_features = emb_dim)
        self.L2 = nn.Linear(emb_dim, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)
        
        #throw in linear here
        ### combined both SSAs using linear ==> then throw in positional encoding on combination
       
        self.PE = PositionalEncoding(dropout = 0.1, dim = emb_dim, pos_emb_type = 'sine')
       
        self.L3 = nn.Linear(emb_dim, emb_dim)
       
        #LSTM layer
        self.LSTM = nn.LSTM(input_size = emb_dim, hidden_size = hidden_size, batch_first = True )
       
        #embedding the demographic info
        self.DEM = Demographics(150, dem_emb_size)
       
        self.linear =  nn.Linear(in_features = hidden_size + dem_emb_size * 2, out_features = 2)
 
    def forward(self, batch):
        prs = self.SSA_proc(batch.proc_cd)[0]
        dis = self.SSA_diag(batch.diag_cd)[0]
       
        #using Irfan-recommended approach
        # ===> this is basically copying a section of code from Jon's repo
        x = F.relu( self.L1(dis) )
        x = x * prs
        x = x + dis
        residual = x
        x = self.L2(x)
        x = F.relu(x) + residual
        x = self.layer_norm(x)
        x = F.relu(self.L3( self.PE(x, dates = batch.dates) ))
      
        
        Lout = self.LSTM(x)[0][:, -1, :]
       
        #stacking on the demographics
        add_dems = torch.cat( (Lout, self.DEM(batch.age, batch.gender)), axis = 1)
       
        #add a linear layer to get to output dimensions
        out = F.softmax( self.linear(add_dems), dim = 1 )
        return(out)
       
 
class MultiHeadedAttention(nn.Module):
   
    """
    head_count (int): number of attention heads
    model_dim (int): this should be equal to the dimension of key,query,value vectors
   
    """
    def __init__(self, head_count,model_dim,dropout=0.1):
        assert model_dim % head_count == 0
       
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super().__init__()
        self.head_count=head_count
        self.linear_keys = nn.Linear(model_dim,head_count * self.dim_per_head,bias=True)
        self.linear_values = nn.Linear(model_dim,head_count * self.dim_per_head,bias=True)
        self.linear_query = nn.Linear(model_dim,head_count * self.dim_per_head,bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim,model_dim,bias=True)
   
    def forward(self, key, value, query, mask=None):
        """
        query (FloatTensor) : batch x seq_len x feat_len x dim
        value (FloatTensor) : batch x seq_len x feat_len x dim
        query (FloatTensor) : batch x seq_len x feat_len x dim\
        mask (LongTensor)   : batch x seq_len x 1 x feat_len
        output (FloatTensor): batch x seq_len x feat_len x dim
        """
       
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        seq_len = key.size(1)
        feat_len = key.size(2)
        query_len = query.size(1)
       
        def shape(x):
            return x.view(batch_size,seq_len,feat_len,head_count,dim_per_head).transpose(2,3)
       
        def unshape(x):
            return x.transpose(2,3).contiguous().view(batch_size,seq_len,feat_len,head_count * dim_per_head)
       
        
        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
       
        key = shape(key)
        value = shape(value)
        query = shape(query)
       
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query,key.transpose(3,4))
        if mask is not None:
            mask=mask.unsqueeze(2) # [B, S, 1, 1, claims]
            scores = scores.masked_fill(mask,-1e18)
       
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
       
        context = unshape(torch.matmul(drop_attn,value))
       
        output = self.final_linear(context)
       
        return output
    def update_dropout(self, dropout):
        self.dropout.p = dropout
 
class MultiheadAttentionVisEncoder(nn.Module):
    def __init__(self,head_count,model_dim,embeddings,dropout=0.1):
        super().__init__()
        self.embeddings=embeddings
        self.selfattn=MultiHeadedAttention(head_count,model_dim,dropout)
        self.layer_norm=nn.LayerNorm(model_dim,eps=1e-6)
        self.dropout=nn.Dropout(dropout)
        self.elu=nn.ELU()
       
    def forward(self,src):
        emb=self.embeddings(src) # [batch_size x seq_len x feat_len x d_model
        emb=self.dropout(self.layer_norm(emb))
        w_batch,w_len,_= src.size()
        padding_idx=self.embeddings.word_padding_idx
        mask=src.data.eq(padding_idx).unsqueeze(2) #[batch_size x src_len x 1 x feat_len]
        lengths=torch.sum(src.data.ne(padding_idx),dim=-1) #[batch_size x src_len]
        attn_out=self.selfattn(emb,emb,emb,mask)
       
        #residual connection and nonlinearity
        out_nl=self.elu(self.dropout(attn_out)+emb)
       
        
        out_masked = torch.mul(1-mask.squeeze(2).unsqueeze(3).float(),out_nl)
        visit_rep = torch.sum(out_masked,dim=2) 
        lengths_filled = lengths.masked_fill(lengths.data.eq(0),1)
        #can add one more non-linearity here
        visit_rep = torch.div(visit_rep,torch.sqrt(lengths_filled.unsqueeze(2).float()))
        penalty=None
       
        return visit_rep, lengths, penalty
 
 
 
class CompleteMHA(nn.Module):
    def __init__(self, emb_dim = 32, hidden_size = 20, dem_emb_size = 2, head_count = 2 ):
        super(CompleteMHA, self).__init__()
       
        
        
        # try emb_dim = 16, hidden size ==> 8
        # hidden at 16 might be small
       
        # keep increasing sizes
 
        #embedding layer for proc tensors
        self.EMB_proc = Embeddings(word_vec_size = emb_dim,
                     word_vocab_size = len( proc_field.vocab ),
                     word_padding_idx = 0)
        #stuctured attention encoder
        #self.SSA_proc = StructuredAttentVisEncoder(emb_dim, emb_dim, num_hops, self.EMB_proc )  
        self.SSA_proc = MultiheadAttentionVisEncoder( head_count = head_count, model_dim = emb_dim,
                                        embeddings = self.EMB_proc)
                                  
        self.EMB_diag = Embeddings(word_vec_size = emb_dim,
                     word_vocab_size = len( visit_field.vocab ),
                     word_padding_idx = 0)
                                  
        #self.SSA_diag = StructuredAttentVisEncoder(emb_dim, emb_dim, num_hops, self.EMB_diag )
       
        self.SSA_diag = MultiheadAttentionVisEncoder( head_count = 4, model_dim = emb_dim,
                                        embeddings = self.EMB_proc)                          
                                   
        self.L1 = nn.Linear(in_features = emb_dim, out_features = emb_dim)
        self.L2 = nn.Linear(emb_dim, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)
       
        #throw in linear here
        ### combined both SSAs using linear ==> then throw in positional encoding on combination
       
        self.PE = PositionalEncoding(dropout = 0.1, dim = emb_dim, pos_emb_type = 'sine')
       
        self.L3 = nn.Linear(emb_dim, emb_dim)
       
        #LSTM layer
        self.LSTM = nn.LSTM(input_size = emb_dim, hidden_size = hidden_size, batch_first = True )
       
        #embedding the demographic info
        self.DEM = Demographics(150, dem_emb_size)
       
        self.linear =  nn.Linear(in_features = hidden_size + dem_emb_size * 2, out_features = 2)
 
    def forward(self, batch):
        prs = self.SSA_proc(batch.proc_cd)[0]
        dis = self.SSA_diag(batch.diag_cd)[0]
       
        #using Irfan-recommended approach
        # ===> this is basically copying a section of code from Jon's repo
        x = F.relu( self.L1(dis) )
        x = x * prs
        x = x + dis
        residual = x
        x = self.L2(x)
        x = F.relu(x) + residual
        x = self.layer_norm(x)
        x = F.relu(self.L3( self.PE(x, dates = batch.dates) ))
      
        
        Lout = self.LSTM(x)[0][:, -1, :]
       
        #stacking on the demographics
        add_dems = torch.cat( (Lout, self.DEM(batch.age, batch.gender)), axis = 1)
       
        #add a linear layer to get to output dimensions
        out = F.softmax( self.linear(add_dems), dim = 1 )
        return(out)