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


def generate_concat_feature(featA_path, featB_path, output_file = None, Return = False):
      featA = load_feature_from_local(featA_path)
      featB = load_feature_from_local(featB_path)
      output_dict = {}
      for ID, value in featA.items():
          if ID in featB:
              output_dict[ID] = np.concatenate([value, featB[ID]], 0)
          else:
              print(f"{ID} failed to map feature in {featB_path}")
      print(f"{len(output_dict)} features concatenated")
      if output_file:
          try:
              with open(output_file, 'wb') as f:
                pickle.dump(output_dict, f)
              print(f'concatenated array saved as {output_file}')
          except:
              print(f'concatenated array failed to save as {output_file}')
              import uuid
              tmp_name = str(uuid.uuid4())
              output_file =os.path.join(os.path.dirname(input_file), tmp_name) 
              with open(output_file, 'wb') as f:
                pickle.dump(output_dict, f)
              print(f'Temp concatenated array saved as {output_file}')
      if Return:
          return output_dict

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

def train_epoch_0(model, train_iterator,loss_function, optimizer, 
              batch_size, 
              max_num_padding, 
              monitor,
              device,
              ground_truth = 'labels'): 
    if isinstance(model, torch.nn.Module):
        model.train()
    loss_cumsum = 0
    all_labels = []
    all_yhat = []
    for batch in train_iterator(batch_size = batch_size, max_num_padding = max_num_padding, device = device):
        y_hat = model(batch)
        assert ground_truth in batch, f"{ground_truth} not in batch dict"
        loss = loss_function(y_hat, batch[ground_truth])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cumsum += loss.item()
        all_yhat.extend(list(np.array(y_hat.squeeze(-1).clone().detach().to('cpu'))))
        all_labels.extend(list(np.array(batch[ground_truth].clone().detach().to('cpu'))))
    loss_cumsum /= len(all_yhat)
    epoch_output = {'Loss':loss_cumsum}
    if monitor:
        epoch_output.update(monitor.compute(all_yhat, all_labels))
    return epoch_output


def train_epoch_clef(model, train_iterator,loss_function, optimizer, 
              batch_size, 
              max_num_padding, 
              monitor,
              device): 
    if isinstance(model, torch.nn.Module):
        model.train()
    loss_cumsum = 0
    all_featA = []
    all_featB = []
    for batch in train_iterator(batch_size = batch_size, max_num_padding = max_num_padding, device = device):
        X, proj_X, proj_feat = model(batch)
        loss = loss_function(proj_X, proj_feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cumsum += loss.item()
        all_featA.extend(list(np.array(proj_X.clone().detach().to('cpu'))))
        all_featB.extend(list(np.array(proj_feat.clone().detach().to('cpu'))))
    loss_cumsum /= len(all_featA)
    epoch_output = {'Loss':loss_cumsum}
    if monitor:
        epoch_output.update(monitor.compute(all_featA, all_featB))
    return epoch_output

def test_clef(model,test_iterator,loss_function, 
              max_num_padding, 
              monitor,
              device,
              return_arr = False): 
    if isinstance(model, torch.nn.Module):
        model.eval()
    loss_cumsum = 0
    all_featA = []
    all_featB = []
    for batch in test_iterator(batch_size = 128, max_num_padding = max_num_padding, device = device):
        with torch.no_grad():
            X, proj_X, proj_feat = model(batch)
            loss = loss_function(proj_X, proj_feat)
            loss_cumsum += loss.item()
        all_featA.extend(list(np.array(proj_X.clone().detach().to('cpu'))))
        all_featB.extend(list(np.array(proj_feat.clone().detach().to('cpu'))))
    loss_cumsum /= len(all_featA)
    test_output = {'Loss':loss_cumsum}
    if monitor:
        test_output.update(monitor.compute(all_featA, all_featB)) 
    if return_arr:
        return test_output, all_featA
    else:
        return test_output

            

def test(model,test_iterator,loss_function, 
              max_num_padding, 
              monitor,
              device,
              ground_truth = 'labels',
              return_arr = False): 
    if isinstance(model, torch.nn.Module):
        model.eval()
    loss_cumsum = 0
    all_labels = []
    all_yhat = []
    for batch in test_iterator(batch_size = 128, max_num_padding = max_num_padding, device = device, test = True, shuffle = False):
        assert ground_truth in batch, f"{ground_truth} not in batch dict"
        with torch.no_grad():
            y_hat = model(batch)
            loss = loss_function(y_hat, batch[ground_truth])
            loss_cumsum += loss.item()
        all_yhat.extend(list(np.array(y_hat.squeeze(-1).clone().detach().to('cpu'))))
        all_labels.extend(list(np.array(batch[ground_truth].clone().detach().to('cpu'))))
    loss_cumsum /= len(all_yhat)
    test_output = {'Loss':loss_cumsum}
    if monitor:
        test_output.update(monitor.compute(all_yhat, all_labels)) 
    if return_arr:
        return test_output, all_yhat
    else:
        return test_output


class trainer:
    def __init__(self, train_epoch, test_epoch = None):
        self.train_epoch = train_epoch
        self.test_epoch = None
        if test_epoch:
            self.test_epoch = test_epoch
    
    def Train(self, model, train_config, test_config,
              num_epoch,
              monitor_score = None,
              multiplier = 1,
              check_point_save = True,
              early_stop_patience = 10,
              log_path = "log"):
                
        best_score = -np.Inf
        best_epoch = 0
        Train_log = {}
        k = 0
        for epoch in range(num_epoch):
            train_config['model'] = model
            train_metrics = self.train_epoch(**train_config)
            if self.test_epoch:
                test_config['model'] = model
                test_metrics = self.test_epoch(**test_config)
            else:
                test_metrics = {}
                  
            # record metrics
            tmp_log = {
              f"train_{key}":round(value, 4) for key, value in train_metrics.items()
            }
            tmp_log.update({
              f"test_{key}":round(value,4) for key, value in test_metrics.items()
            })
            print(f'{epoch + 1}/{num_epoch} epoch : {tmp_log}')
            for col in ['Epoch'] + list(tmp_log.keys()):
                if col not in Train_log:
                    Train_log[col] = []
                else:
                    row = epoch if col == 'Epoch' else tmp_log[col]
                    Train_log[col].append(row)
            
            #record best score
            if epoch + 1 >= early_stop_patience or epoch + 1 == num_epoch:
                if monitor_score not in train_metrics or monitor_score not in test_metrics:
                    if k == 0:
                        print(f'''monitor_score: [{monitor_score}] not in both train and test metrics, using [train_Loss]''')
                        monitor_score, multiplier = 'Loss', -1
                        k = 1
                    score = float(multiplier) * train_metrics[monitor_score]
                else:
                    score = float(multiplier) * np.sqrt(train_metrics[monitor_score] * 
                                        test_metrics[monitor_score])
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_params = model.state_dict() if check_point_save else None
        #output training log
        if log_path: 
          if not os.path.exists(log_path):
              os.mkdir(log_path)
          df = pd.DataFrame(Train_log)
          output_path = os.path.join(log_path, "training_log.csv")
          try:
              df.to_csv(output_path)
              print(f"Save log at {output_path}")
          except:
              print(f"Failed to save log at {output_path}")
        #output parameters
        if check_point_save:
          check_path = log_path if log_path else 'tmp_checkpoint'
          if not os.path.exists(check_path):
              os.mkdir(check_path)
          output_path = os.path.join(check_path, "checkpoint_params.pt")
          try:
              torch.save(best_params, output_path)
              print(f"Save params at {output_path}")
          except:
              print(f"Failed to save params at {output_path}")
        best_score /= multiplier
        return {'Best_score':round(best_score, 4), 'Train_log':Train_log, 'Best_epoch':best_epoch}


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
    header = '_'.join(filename.split('_')[:-3])
    for file in os.listdir(dirpath):
        if f'_{tag}_' in file and header == '_'.join(file.split('_')[:-3]):
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
                
                
def set_seed(seed):
    import random
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
   
    
def get_feature_dim(path):
    x = load_feature_from_local(path)
    tmp = None
    for key, value in x.items():
        if not tmp:
            tmp = value.shape
        elif tmp != value.shape:
            tmp = None
            break
        else:
            tmp = value.shape
    return tmp
