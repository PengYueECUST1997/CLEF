import numpy as np
import pickle
import torch
import torch.nn as nn
import os
import pandas as pd

def load_feature_from_local(feature_path):
    '''
    load feature dict from local path (Using pickle.load() or torch.load())
    the dictionary is like:
        {
          Protein_ID : feature_array [a 1D numpy array]
        }
    '''
    # Try pickle.load() function 
    try:
        with open(feature_path, 'rb') as f:
            obj = pickle.load(f)
        print("File is loaded using pickle.load()")
        return obj
    except (pickle.UnpicklingError, EOFError):
        pass

    # Try torch.load() function
    try:
        obj = torch.load(feature_path)
        print("File is loaded using torch.load()")
        return obj
    except (torch.serialization.UnsupportedPackageTypeError, RuntimeError):
        pass

    print("Unable to load file.")
    return None



class Metrics:
    def __init__(self, metrics_dict):
        self.metrics = metrics_dict
    
    def compute(self, all_yhat, all_labels, flexible_cutoff = False):
        output_score = {}
        for key, method in self.metrics.items():
            output_score[key] = method(all_yhat, all_labels, flexible_cutoff)
        return output_score
        
  


def Predict_0(input_file, 
              Dataset_build, 
              model,
              output_file = None,
              transform_method = None,
              transform_config = None,
              params_path = None,
              loader_config = None,
              binary_threshod = 0.5,
              excel_format = True):
    if transform_method:
       print(f'Transform feature from {input_file}')
       import uuid
       tmp_file = str(uuid.uuid4())+'_tmp'
       tmp_file = os.path.join(os.path.dirname(input_file), tmp_file)
       transform_config = {'input_file':input_file} if not transform_config else transform_config
       try:
          transform_config['Return'] = False
          transform_config['output_file'] = tmp_file if 'output_file' not in transform_config else transform_config['output_file']
          transform_method(**transform_config)
       except:
          transform_config ={
            'input_file':input_file, 'output_file':tmp_file, 'Return':False
          }
          transform_method(**transform_config)
       Predictset = Dataset_build({'feature':tmp_file})
       try: 
          print(f'Remove the temp feature file {tmp_file}')
          os.remove(tmp_file)
       except:
          pass
    else:
      print(f'Direct load feature file from {input_file}')
      Predictset = Dataset_build({'feature':input_file})
    
    Predictset.test_range = range(len(Predictset))
    
    
    if params_path:
      print(f"Load model weights from {params_path}")
      try:
        model.load_state_dict(torch.load(params_path))
      except:
        print(f"Failed to load model weights from {params_path}")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    try:
      loader_config['batch_size'] = 1
      loader_config['shuffle'] = False
      loader_config['test'] = True
      loader_config['device'] = device
    except:
      print(f"Loader configuration {loader_config} is invalid")
      loader_config = {'batch_size':1, 'shuffle':False, 'test':True, 'device':device}
    
    if isinstance(model, torch.nn.Module):
        model.eval()
    result = {x:[] for x in ('ID', 'score', 'label')}
    for batch in Predictset.Dataloader(**loader_config):
        with torch.no_grad():
            y_hat = model(batch)
        result['score'].extend(list(np.array(y_hat.squeeze(-1).clone().detach().to('cpu'))))
        result['ID'].extend(batch['ID'])
        
    result['label'] = ['Yes' if x > binary_threshod else 'No' for x in result['score']]
    
    if output_file:
        import pandas
        df = pd.DataFrame(result)
        if excel_format:
            filename = os.path.split(output_file)[-1]
            filename = '.'.join(filename.split('.')[:-1]+['xlsx'])
            output_file = os.path.join(os.path.dirname(output_file), filename)
            df.to_excel(output_file)
        else:
            df.to_csv(output_file, sep = '\t')
        print(f"Save predict result at {output_file}")
    return result

def assign_labels(sample_ids, tag_label_mapping):
    '''
    iterate a list of sample ids, generate a ID-label dict by the given tag_label mapping dict
    label is assigned by discriminating whether the identifier string tag appears in ID string
    the ID not mapping any tag will be assigned 0 by default
      the dictionary is like:
      {
        Protein_ID : label [integer]
      }
    '''
    labeled_dict = {}
    for sample_id in sample_ids:
        
        apperance = np.array([x in sample_id for x in tag_label_mapping]) 
        if apperance.sum() > 0: # discriminate tag appear in sample id
            label = list(tag_label_mapping.values())[
                np.where(apperance == 1)[0][-1] # in case more than one tag appear in ID, only the last was consider
                ]
        else:
            label = 0
        labeled_dict[sample_id] = label

    return labeled_dict


def Feature_concatenate(*feature_dicts):
    '''
    merge / combine input features by concatenation
    the input dictionary is like:
        {
        Protein_ID : feature_array [a 1D numpy array]
        }
    '''
    merged_dict = {}
    max_dim = 0
    for d in feature_dicts:
        for key, value in d.items():
            if key in merged_dict:
                merged_dict[key] = np.concatenate([merged_dict[key], value], 0)
            else:
                merged_dict[key] = value
            max_dim = merged_dict[key].shape[0]  if max_dim < merged_dict[key].shape[0] else max_dim
    output_dict = {}
    for key, value in merged_dict.items():
        if value.shape[0] == max_dim:  # only merged value was output
            output_dict[key] = value 
    return output_dict
  
def generate_CLEF_feature(model,input_embeddings_path, output_file = "temp", maxlen = 254, mlp_projection = True, device = "cpu"):
    model.eval()
    with open(input_embeddings_path, 'rb') as f:
        input_embedding = pickle.load(f)
    
    output_dict = {}
    
    for key, value in input_embedding.items():

        input_feat = value.mean(0)   
        seq_input = torch.from_numpy(value[:maxlen + 2, :]).to(torch.float32)
        valid_lens = torch.tensor([min(seq_input.shape[0], maxlen + 2)]).to(torch.long) 
        if seq_input.shape[0] < maxlen + 2:
            num_pads = maxlen - seq_input.shape[0] + 2
            padding = torch.zeros([num_pads, seq_input.shape[1]])
            seq_input = torch.cat([seq_input, padding], 0)  
            assert seq_input.shape[0] == maxlen + 2
        seq_input = seq_input.unsqueeze(0).to(device)
        valid_lens = valid_lens.to(device)
        with torch.no_grad():
            encoded_feature, projected_feature= model(seq_input, valid_lens, features_input = None)
        if mlp_projection:
            output_feat = projected_feature.squeeze(0)
        else:
            output_feat = encoded_feature.squeeze(0)
        output_feat = np.array(output_feat.detach().to('cpu'))
        output = output_feat
        output_dict[key] = output
    with open(output_file, 'wb') as f:
        pickle.dump(output_dict, f)
        
        
def generate_ESM_feature(input_embeddings_path, output_file = "temp"):
    
    with open(input_embeddings_path, 'rb') as f:
        input_embedding = pickle.load(f)
    
    output_dict = {}
    
    for key, value in input_embedding.items():

        output_feat = value.mean(0)   
        output = output_feat
        output_dict[key] = output
        
    with open(output_file, 'wb') as f:
        pickle.dump(output_dict, f)

def auto_make_traintest_config(input_path, align = True):
    filename = os.path.split(input_path)[-1]
    dirpath = os.path.dirname(input_path)
    tag = filename.split('_')[-2] if 'esm_rep' not in filename else filename.split('_')[-3]
    train_data_config, test_data_config = {}, {}
    for file in os.listdir(dirpath):
        if f'_{tag}_' in file:
            feat_type = file.split('_')[-1] if 'esm_rep' not in file else file.split('_')[-2]
            data_type = file.split('_')[-3] if 'esm_rep' not in file else file.split('_')[-4]
            if data_type.lower() == 'test':
                test_data_config[feat_type] = os.path.join(dirpath, file)
            elif data_type.lower() == 'train':
                train_data_config[feat_type] = os.path.join(dirpath, file)
    if align:
        tmp_train, tmp_test = {}, {}
        for key, value in train_data_config.items():
            if key in test_data_config:
                tmp_train[key] = value
                tmp_test[key] = test_data_config[key]
            else:
                print(f"{key} tag feature not in train and test")
        train_data_config, test_data_config = tmp_train,tmp_test
        
    return train_data_config, test_data_config



def switch_label(input_file, tag_label_mapping):
    x = load_feature_from_local(input_file)
    y = assign_labels(list(x.keys()), tag_label_mapping)
    with open(input_file, 'wb') as f:
        pickle.dump(y, f)

def auto_switch_binary_label(input_path, sele):
    filename = os.path.split(input_path)[-1]
    dirpath = os.path.dirname(input_path)
    tag = filename.split('_')[-2] if 'esm_rep' not in filename else filename.split('_')[-3]
    tag_label_mapping = {sele:1}
    for file in os.listdir(dirpath):
        if f'_{tag}_' in file:
            feat_type = file.split('_')[-1] if 'esm_rep' not in file else file.split('_')[-2]
            if feat_type.lower() == 'label':
                switch_label(os.path.join(dirpath, file), tag_label_mapping)
                print(f"{file} label switched as {sele}")
                
                
    
    
