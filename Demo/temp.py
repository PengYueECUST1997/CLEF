import numpy as np
import torch.nn as nn

from CLEF import clef
from utils import generate_CLEF_feature, generate_ESM_feature, assign_labels

max_seq_len = 128
encoder_parmas = '../Notebooks/Visual/PSSM_DPC/encoder_params'
#encoder_parmas ='../Notebooks/Visual/Biobert/encoder_params'
input_esm_list = [
  "./AMP/AMP_test_esm(128)"
  ]
#input_esm_list = ["../Benchmark-partB/DeepSecE_train_esm(256)", "../Benchmark-partB/DeepSecE_test_esm(256)"]
device='cuda:0'
model=clef(1280, 400, mlp_relu = True, feat_mlp_relu = True, feature_norm = True).to(device)
model.load_state_dict(torch.load(encoder_parmas))
log_path = "../Tutorial/AMP"
try:
  os.mkdir(log_path)
except:
  pass
tag_label_mapping = {
  '|anti':1
}
initial = True
tag = 'DPC'
esm_path = input_esm_list[0]
for esm_path in input_esm_list:
  
  header = os.path.split(esm_path)[-1]
  header = '_'.join(header.split('_')[:-1])+f'_{tag}_clef'
  input_embeddings_path = esm_path
  output_file = os.path.join(log_path, header)
  generate_CLEF_feature(model, input_embeddings_path, output_file, maxlen = 256, mlp_projection = False, device = device)
  if initial:
      header = os.path.split(esm_path)[-1]
      header = '_'.join(header.split('_')[:-1])+f'_{tag}_esm_rep'
      input_embeddings_path = esm_path
      output_file = os.path.join(log_path, header)
      generate_ESM_feature(input_embeddings_path, output_file)
      x = pickle.load(open(output_file, "rb"))
      header = os.path.split(esm_path)[-1]
      header = '_'.join(header.split('_')[:-1])+f'_{tag}_label'
      output_file = os.path.join(log_path, header)
      sample_ids = list(x.keys())
      labeled_dict = assign_labels(sample_ids, tag_label_mapping)
      pickle.dump(labeled_dict, open(output_file, 'wb'))


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
        assert ground_truth in batch f"{ground_truth} not in batch dict"
        loss = loss_function(y_hat.squeeze(-1), batch[ground_truth].to(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cumsum += loss.item()
        all_yhat.extend(list(np.array(y_hat.squeeze(-1).clone().detach().to('cpu'))))
        all_labels.extend(list(np.array(batch[ground_truth].clone().detach().to('cpu'))))
    epoch_output = {'Loss':loss_cumsum}
    if monitor:
        epoch_output.update(monitor.compute(all_yhat, all_labels))
    return epoch_output


def test(model,test_iterator,loss_function
              max_num_padding, 
              monitor,
              device,
              ground_truth = 'labels'): 
    if isinstance(model, torch.nn.Module):
        model.eval()
    loss_cumsum = 0
    all_labels = []
    all_yhat = []
    for batch in test_iterator(batch_size = 128, max_num_padding = max_num_padding, device = device, test = True, shuffle = False):
        assert ground_truth in batch f"{ground_truth} not in batch dict"
        with torch.no_grad():
            y_hat = model(batch)
            loss = loss_function(y_hat.squeeze(-1), batch[ground_truth].to(torch.float32))
            loss_cumsum += loss.item()
        all_yhat.extend(list(np.array(y_hat.squeeze(-1).clone().detach().to('cpu'))))
        all_labels.extend(list(np.array(batch[ground_truth].clone().detach().to('cpu'))))
    test_output = {'Loss':loss_cumsum}
    if monitor:
        test_output = monitor.compute(all_yhat, all_labels)
    return test_output

import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
import os
import pandas as pd
from Data_utils import Potein_rep_datasets
from Module import test_dnn, BCELoss, ContrastiveLoss
from utils import Metrics, Train, assign_labels
from Metrics_method import compute_acc, compute_f1, compute_mcc



x = pickle.load(open("./AMP/bastionX/DeepSecE_test_bastionX_esm", "rb"))
print(next(iter(x.items())))
y = assign_labels(list(x.keys()), tag_label_mapping = {'Non-':0, 'T1SE':1, "T2SE":2,"T3SE":3,"T4SE":4, "T6SE":5})
y = {key.split('~')[0]:value for key, value in x.items()}
pickle.dump(y, open("./AMP/bastionX/DeepSecE_test_bastionX_label", "wb"))

data_path_config = {'esm_feature':"../Benchmark-partB/DeepSecE_train_esm(512)",
                    "B_feature":"../Benchmark-partB/BastionX/CLEF_feature/DeepSecE_train_bastionX_ori"}
Dataset = Potein_rep_datasets(data_path_config)

import torch
from CLEF import clef
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = clef(640, 5).to(device)
import torch.optim as optim
import os
lr = 0.00002
optimizer =optim.Adam(model.parameters(), lr=lr)
loss_function= ContrastiveLoss()
batch_size = 128
# max_num_padding = None
monitor_score = None
train_iterator = Dataset.Dataloader
test_iterator = Dataset.Dataloader
monitor = Metrics(metrics_dict = {})
num_epoch = 15
monitor_score = None

from utils import test_clef, train_epoch_clef, trainer
train_config = {
  'train_iterator':Dataset.Dataloader,
  'loss_function':loss_function, 
  'optimizer':optimizer, 
  'batch_size':128, 
  'max_num_padding':512, 
  'monitor':monitor,
  'device':device
}
test_config = {}

self = trainer(train_epoch = train_epoch_clef)        
self.Train(model, num_epoch=num_epoch, train_config=train_config, test_config={},early_stop_patience=30)


from utils import load_feature_from_local
input_file = "../Benchmark-partB/DeepSecE_test_esmpool(512)"
x = load_feature_from_local(input_file)
y = {}
for key, value in x.items():
    ID = key
    if key.split('~')[-1] in [str(x) for x in range(10)]:
        ID = key.split('~')[0]
    y[ID] = value
    print(ID)
pickle.dump(y, open(input_file, "wb"))
next(iter(x.keys()))

from Feature_transform  import generate_clef_feature
input_file = "../Benchmark-partB/DeepSecE_train_esm(512)"
tag = 'bastionX'
dirname = f"./AMP/{tag}"
if not os.path.exists(dirname):
    os.mkdir(dirname)
header = '_'.join(os.path.split(input_file)[-1].split('_')[:-1])
output_file = os.path.join(dirname, f"{header}_{tag}_clef")
model = clef(640, 5)
params_path = "./log/checkpoint_params.pt"
loader_config = {'batch_size':64, 'max_num_padding':512}
config = {
  'input_file':input_file,
  'output_file':output_file,
  'model':model,
  'params_path':params_path,
  'loader_config':loader_config
}
generate_clef_feature(**config)

input_embeddings_path = "../Benchmark-partB/DeepSecE_train_esm(512)"
header = '_'.join(os.path.split(input_embeddings_path)[-1].split('_')[:-1])
from utils import generate_ESM_feature
conf = {
'input_embeddings_path' : input_embeddings_path,
'output_file' : os.path.join(dirname, f"{header}_{tag}_esm"),
}
generate_ESM_feature(**conf)

from utils import auto_make_traintest_config, auto_switch_binary_label, assign_labels
input_path = os.path.join(dirname, f"{header}_{tag}_clef")
file = f"{header}_{tag}_clef"
sele = 'T3'
data_type = file.split('_')[-3] if 'esm_rep' not in file else file.split('_')[-4]
k, l = ('test', 'train') if data_type == 'test' else ('train', 'test')
label_dict = assign_labels((pickle.load(open(input_path, "rb"))).keys(), {sele:1})
pickle.dump(label_dict, open(os.path.join(dirname, f"{header}_{tag}_label"), "wb"))
label_dict = assign_labels((pickle.load(open(input_path.replace(k, l), "rb"))).keys(), {sele:1})
pickle.dump(label_dict, open(os.path.join(dirname, f"{header}_{tag}_label".replace(k, l)), "wb"))




import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
import os
import pandas as pd
from Data_utils import Potein_rep_datasets
from Module import test_dnn, BCELoss
from utils import Metrics, Train
from Metrics_method import compute_acc, compute_f1, compute_mcc




from utils import auto_make_traintest_config, auto_switch_binary_label, assign_labels
import torch.optim as optim
sele = 'T4'
input_path = "./AMP/bastionX/DeepSecE_test_bastionX_clef"

tag = os.path.split(input_path)[-1].split('_')[-2]
output_file = f"{tag}_compare_log_{sele}.csv"
initial_model = test_dnn
train_data_config, test_data_config = auto_make_traintest_config(input_path, align = False)

auto_switch_binary_label(input_path, sele)

num_trails = 3
feat_dim = 5

monitor = Metrics({'Accuracy':compute_acc, 'F1':compute_f1, "MCC":compute_mcc})
Train_config = {'loss_function':BCELoss(), 
                  'batch_size':32, 
                  'num_epoch':55, 
                  'max_num_padding':None,
                  'monitor':monitor,
                  'monitor_score':'Accuracy',
                  'early_stop_patience':30}

def Feature_benchmark(train_data_config, test_data_config,
                      initial_model,
                      feat_dim,
                      Train_config,
                      model_config = {'num_embeds':1280, 'finial_drop':0.5},
                      lr = 0.000008,
                      initial_optimizer = optim.Adam,
                      esm_dim = 640,
                      num_trails = 4,
                      label_tag = 'label',
                      train_range = None,
                      test_range = None,
                      seed = 666,
                      flexible_cutoff = True,
                      output_file = "./tmp_compare_log.csv"
                      ):
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    test_candidate = [x for x in train_data_config  if x != label_tag]
    model_param_config = {'esm':esm_dim, 'clef':esm_dim, 'ori':feat_dim, 'concat':feat_dim+esm_dim, 'clefcat':esm_dim+esm_dim}
    
    output = pd.DataFrame({})
    
    for trail in range(num_trails):
        compare_log = {}
        for mode in test_candidate:
            tmp_train_config = {'feature':train_data_config[mode], 'label':train_data_config[label_tag]}
            Dataset = Potein_rep_datasets(tmp_train_config, label_tag = label_tag)
            if train_range:
                Dataset.train_range = train_range
            if test_data_config:
                tmp_test_config = {'feature':test_data_config[mode], 'label':test_data_config[label_tag]}
                testset = Potein_rep_datasets(tmp_test_config, label_tag = label_tag)
                if test_range:
                    testset.test_range = test_range
            else:
                Dataset.split_test(0.1)
                testset = Dataset
            model_config['num_embeds'] = model_param_config[mode]
            model = initial_model(**model_config).to(device)
            optimizer = initial_optimizer(model.parameters(), lr)
            
            Train_config.update({'model':model, 'optimizer':optimizer, 
                                'train_iterator':Dataset.Dataloader,
                                'test_iterator':testset.Dataloader,
                                'device':device})
            tmp_result = Train(**Train_config)
            best_score, train_log, best_epoch = list(tmp_result.values())
            tmp_log = {'best_score':best_score}
            for key, value in train_log.items():
                if key not in  ('Epoch','train_Loss','test_Loss'):
                    tmp_log[key] = value[best_epoch - 1]
            print(f"{mode}:{tmp_log}\n")   
            compare_log[mode] = tmp_log
        board = {'mode': list(compare_log.keys())}
        board.update({x:[compare_log[i][x] for i in list(compare_log.keys())] for x in list(tmp_log.keys())})
        print(pd.DataFrame(board))
        output = pd.concat([output, pd.DataFrame(board)], 0)
    output.to_csv(output_file, index = False)
    

from Feature_transform import fasta_to_EsmRep
conf = {
'input_fasta' : "../Notebooks/Root/DeepSecE_train.faa",
'output_file' : "../Benchmark-partB/DeepSecE_train_esm(512)",
'maxlen' : 512,
'Return' : False,
'Final_pool' : False,
'pretrained_model_params':'../pretained_model/esm2_t30_150M_UR50D.pt'
}
fasta_to_EsmRep(**conf)
fasta_to_EsmRep(input_fasta, output_file = None, 
                      pretrained_model_params = '../pretained_model/esm2_t33_650M_UR50D.pt',
                      maxlen = 256,
                      Return = True, 
                      Final_pool = False)



from Module import test_dnn
import os
from utils import auto_make_traintest_config, auto_switch_binary_label, assign_labels, Metrics, Train
import torch.optim as optim
import torch
from Data_utils import Potein_rep_datasets
import torch.nn as nn
import numpy as np
from Metrics_method import compute_acc_multi
import pandas as pd
input_path = "./AMP/bastionX/DeepSecE_test_bastionX_clef"
sele = 'multi'
tag = os.path.split(input_path)[-1].split('_')[-2]
output_file = f"{tag}_compare_log_{sele}.csv"
initial_model = test_dnn
train_data_config, test_data_config = auto_make_traintest_config(input_path, align = False)
num_trails = 3
feat_dim = 5
esm_dim = 640
num_class = 6
label_tag = 'label'
device = 'cuda:0' if torch.cuda.is_available() else "cpu"
train_range, test_range = None, None
test_candidate = [x for x in train_data_config  if x != label_tag]
model_param_config = {'esm':esm_dim, 'clef':esm_dim, 'ori':feat_dim, 'concat':feat_dim+esm_dim, 'clefcat':esm_dim+esm_dim}
model_config = {'num_embeds':640, 'finial_drop':0.5}
initial_optimizer = optim.Adam
lr = 0.00003
model_config['num_embeds'] = model_param_config[mode]
model_config['out_dim'] = num_class
model = initial_model(**model_config).to(device)
optimizer = initial_optimizer(model.parameters(), lr)
loss_function = nn.CrossEntropyLoss()
batch_size = 32
num_epoch = 65
max_num_padding = None
train_iterator = Dataset.Dataloader
test_iterator = testset.Dataloader
monitor_score = 'Accuracy'
early_stop_patience = 50
Train_config = {'loss_function':loss_function, 
                  'batch_size':batch_size, 
                  'num_epoch':num_epoch, 
                  'max_num_padding':max_num_padding,
                  'monitor':monitor,
                  'monitor_score':monitor_score,
                  'early_stop_patience':early_stop_patience}
Train_config.update({'model':model, 'optimizer':optimizer, 
'train_iterator':Dataset.Dataloader,
'test_iterator':testset.Dataloader,
'device':device})
monitor =  Metrics({'Accuracy':compute_acc_multi})


def Feature_benchmark(train_data_config, test_data_config,
                      initial_model,
                      feat_dim,
                      Train_config,
                      model_config = {'num_embeds':1280, 'finial_drop':0.5},
                      lr = 0.000008,
                      initial_optimizer = optim.Adam,
                      esm_dim = 640,
                      num_trails = 4,
                      label_tag = 'label',
                      train_range = None,
                      test_range = None,
                      seed = 666,
                      flexible_cutoff = True,
                      output_file = "./tmp_compare_log.csv"
                      ):
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    test_candidate = [x for x in train_data_config  if x != label_tag]
    model_param_config = {'esm':esm_dim, 'clef':esm_dim, 'ori':feat_dim, 'concat':feat_dim+esm_dim, 'clefcat':esm_dim+esm_dim}
    
    output = pd.DataFrame({})
    
    for trail in range(num_trails):
        compare_log = {}
        for mode in test_candidate:
            tmp_train_config = {'feature':train_data_config[mode], 'label':train_data_config[label_tag]}
            Dataset = Potein_rep_datasets(tmp_train_config, label_tag = label_tag)
            if train_range:
                Dataset.train_range = train_range
            if test_data_config:
                tmp_test_config = {'feature':test_data_config[mode], 'label':test_data_config[label_tag]}
                testset = Potein_rep_datasets(tmp_test_config, label_tag = label_tag)
                if test_range:
                    testset.test_range = test_range
            else:
                Dataset.split_test(0.1)
                testset = Dataset
            model_config['num_embeds'] = model_param_config[mode]
            model = initial_model(**model_config).to(device)
            optimizer = initial_optimizer(model.parameters(), lr)
            
            Train_config.update({'model':model, 'optimizer':optimizer, 
                                'train_iterator':Dataset.Dataloader,
                                'test_iterator':testset.Dataloader,
                                'device':device})
            tmp_result = Train(**Train_config)
            best_score, train_log, best_epoch = list(tmp_result.values())
            tmp_log = {'best_score':best_score}
            for key, value in train_log.items():
                if key not in  ('Epoch','train_Loss','test_Loss'):
                    tmp_log[key] = value[best_epoch - 1]
            print(f"{mode}:{tmp_log}\n")   
            compare_log[mode] = tmp_log
        board = {'mode': list(compare_log.keys())}
        board.update({x:[compare_log[i][x] for i in list(compare_log.keys())] for x in list(tmp_log.keys())})
        print(pd.DataFrame(board))
        output = pd.concat([output, pd.DataFrame(board)], 0)
    output.to_csv(output_file, index = False)
