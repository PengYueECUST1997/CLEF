import torch
import torch.nn as nn
import pandas as pd
<<<<<<< HEAD
=======
import numpy as np
>>>>>>> fe42717cb74680de2878cc0abe040b030b85d61f

class trainer:
    def __init__(self, train_epoch, test_epoch = None):
        self.train_epoch = train_epoch
        self.test_epoch = None
        if test_epoch:
            self.test_epoch = None
    
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

def Train(model, train_iterator,test_iterator, loss_function, optimizer, 
              batch_size, num_epoch,
              max_num_padding, 
              monitor,
              device,
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
        train_metrics = train_epoch_0(model, train_iterator,
              loss_function, 
              optimizer, 
              batch_size, 
              max_num_padding, 
              monitor,
              device)
        test_metrics = test(model,test_iterator,loss_function,
              max_num_padding, 
              monitor,
              device)
              
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

