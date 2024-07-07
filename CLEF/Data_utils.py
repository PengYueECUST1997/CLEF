import numpy as np
import pickle
import torch
from sklearn.model_selection import train_test_split
import random
from utils import load_feature_from_local

class Potein_rep_datasets:
  
  def __init__(self, input_path, train_range = None, test_range = None, label_tag = 'label'):
        '''
        [input_path] is a Path_dict containing feature ID and corresponding Local_path
        e.g {'feature':'./path/to/you/feature_file'}
        '''
        sequence_data = {}
        try:
            for key, value in input_path.items():
                if isinstance(value, str):
                    print(f"try to load feature from path:{value}")
                    tmp = load_feature_from_local(value)
                elif isinstance(value, np.ndarray):
                    print(f"try to load feature from numpy_array")
                    tmp = value
                else:
                    print(f"can not load feature {key}")
                    continue
                for ID, feat in tmp.items():
                    if ID not in sequence_data:
                        sequence_data[ID] = {key:feat}
                    else:
                        sequence_data[ID].update( {key:feat} )
            
            self.sequence_data = {}   
            for key, value in sequence_data.items():
                if len(value) < len(input_path):
                   print(f"imcomplete feature ID {key} removed")
                else:
                   self.sequence_data[key] = value
            
            if label_tag not in input_path:
                print(f"Add mock label [{label_tag}] of 0 for each sample")
                for key in self.sequence_data:
                    self.sequence_data[key][label_tag] = 0
        except:
            print(f"No valid [input_path] to load : {input_path}, return an empty dataset")
            self.sequence_data = {}
               
        self.data_indices = {i : ID for i, ID in enumerate(self.sequence_data)}
        
        self.label_tag = label_tag
        
        print(f"total {len(self.data_indices)} sample loaded")
        self.feature_list = list(input_path.keys())
        
        self.train_range = train_range
        self.test_range = test_range
        
        if not self.train_range:
            self.train_range = range(len(self.data_indices))
      
        if not self.test_range:
            self.test_range = range(len(self.data_indices))
            
        
        
        
  def Dataloader(self, batch_size, shuffle = True, 
                          test = False,
                          max_num_padding = None,
                          device = 'cpu'):

        sele_range = self.test_range if test else self.train_range
        Nsample=len(list(sele_range))
        indices=list(sele_range)
        if shuffle:
            random.shuffle(indices)
        datasets = []
        IDs = []
        n = 0
        for i in indices:
            n += 1
            
            ID = self.data_indices[i]
            IDs.append(ID)
            data = self.sequence_data[ID]
            datasets.append(data)
            
            if len(datasets) == batch_size or n == Nsample:
                try:
                    labels = torch.tensor([x[self.label_tag] for x in datasets]).to(torch.long)
                except:
                    print(f"feature <{self.label_tag}> is not a valid label value, using mock labels instead.")
                    labels =  torch.tensor([0 for x in datasets]).to(torch.long)
                batch = {
                  'labels':labels.to(device),
                  'ID':IDs
                }
                for key in self.feature_list:
                    if key != self.label_tag:
                      if max_num_padding:
                          padded_seq_input = [pad_to_max_length(x[key], max_num_padding)[0] for x in datasets]
                          valid_lens = torch.Tensor([pad_to_max_length(x[key], max_num_padding)[1] for x in datasets ]).to(torch.long)
                          seq_input = np.concatenate([np.expand_dims(x, axis = 0) for x in padded_seq_input])
                          seq_input = torch.from_numpy(seq_input).to(torch.float32)
                          batch.update({key:seq_input.to(device)})
                          if (valid_lens.max() > 0).item():
                              if 'valid_lens' in batch :
                                  try:
                                      assert (batch['valid_lens'] == valid_lens).sum().item() == batch_size 
                                  except:
                                      print(f"Warning: please make sure valid lens of 2D tensors is the same \n{batch['valid_lens']}\n{valid_lens}")
                            
                              batch.update({'valid_lens':valid_lens.to(device)})
                      else:
                          seq_input = np.vstack([x[key] for x in datasets])
                          seq_input = torch.from_numpy(seq_input).to(torch.float32)
                          batch.update({key:seq_input.to(device)})
                
                datasets = []
                IDs = []
                
                yield batch
    
  def __len__(self):
        return len(self.sequence_data)
      
  def split_test(self, test_size = 0.1):
    
        if len(self.sequence_data) > 1:
            train_indices, test_indices = train_test_split(range(len(self.sequence_data)), test_size=test_size, random_state=42)
            self.train_range = train_indices
            self.test_range = test_indices
        else:
            print(f"Number of {len(self)} data can not be splited.")
        
  
class PPI_rep_datasets:
    '''
    load [1D sequence_representation numpy array, pairwise PPI indeces] and yield tensor for training.
    sequence_data---a dict containing feature of each sequence
        {protein_ID :  feature (a 1D numpy array)}.
    PPI_indeces---a dict containing pairwise protein indeces
    '''
    def __init__(self, rep_path, index_path,  train_range = None, test_range = None):
        self.sequence_data = pickle.load(open(rep_path, 'rb'))
        self.PPI_indeces = pickle.load(open(index_path, 'rb'))
        self.data_indices = self.PPI_indeces
        
        self.train_range = train_range
        self.test_range = test_range
        
        if not self.train_range:
            self.train_range, test_indices = train_test_split(range(len(self.data_indices)), test_size=0.1, random_state=42)
        if not self.test_range:
            self.test_range = test_indices
        
    def Dataloader(self, batch_size, shuffle = True, 
                          test = False,
                          device = 'cpu'):

        sele_range = self.test_range if test else self.train_range
        Nsample=len(list(sele_range))
        indices=list(sele_range)
        if shuffle:
            random.shuffle(indices)
            
        datasets = []
        IDs = []
        n = 0
        for i in indices:
            n += 1
            ID = '-'.join(self.data_indices[i])
            IDs.append(ID)
            embedding_A = self.sequence_data[self.data_indices[i][0]]
            embedding_B = self.sequence_data[self.data_indices[i][1]]
            dataset = (embedding_A, embedding_B)
            datasets.append(dataset)
            
            if len(datasets) == batch_size or n == Nsample:
                
                seq_input_A = torch.cat([torch.from_numpy(x[0]).unsqueeze(0) for x in datasets], 0).to(torch.float32)
                seq_input_B = torch.cat([torch.from_numpy(x[1]).unsqueeze(0) for x in datasets], 0).to(torch.float32)
                idx = IDs
                datasets = []
                IDs = []
                
                batch = {
                  'A':seq_input_A.to(device),
                  'B':seq_input_B.to(device),
                  'ID':idx
                }
                yield batch
    
    def __len__(self):
        return len(self.data_indices)

        


def pad_to_max_length(seq, max_length, len_dim = 0, feat_dim = 1):
    if len(seq.shape) < 2:
        return seq, 0
    seq_length = seq.shape[len_dim]
    if seq_length < max_length:
        padded_seq = np.zeros([max_length, seq.shape[feat_dim]])
        padded_seq[:seq_length] = seq
        return padded_seq, seq_length
    else:
        return seq[:max_length, :], max_length
