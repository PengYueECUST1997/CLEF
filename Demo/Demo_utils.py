
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys

def find_root_path():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    return project_root

src_path = os.path.join(os.path.join(find_root_path(), "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from Data_utils import Potein_rep_datasets
from utils import *
from Feature_transform import *
from CLEF import *
from Module import test_dnn



    
    

def predict_from_1D_rep(input_file, initial_model, params_path,
                        output_file = None,cutoff = None,
                        Return = True):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_hidden = check_hidden_layer_dimensions(load_feature_from_local(input_file, silence=True))
    assert num_hidden, f"Dimension numbers of the last dimension is not same; {input_file}"

    model = initial_model(num_hidden).to(device)
    eff_type = os.path.split(params_path)[-1].lower().split('classifier')[0].split('-')[-1]
    eff_type = f'{eff_type.upper()}SE' if eff_type in ['t3', 't4', 't6'] else 'Effector'
    Dataset = Potein_rep_datasets({'feature':input_file})
    if not cutoff:
        try:
            cutoff = float(os.path.split(params_path)[-1].lower().split('cutoff')[0].split('-')[-1])
        except:
            cutoff = 0.5
    print(f'Binary cutoff of {cutoff} used.')
    output = {
        'ID':[],
        'pred':[],
        eff_type:[]
    }
    model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
    model.eval()
    Dataset.test_range = range(len(Dataset))
    for batch in Dataset.Dataloader(batch_size=32,shuffle=False,max_num_padding=None,test=True,device=device):
        with torch.no_grad():
            y_pred = model(batch)
        y_pred = y_pred.detach().to('cpu').numpy()
        output['ID'].extend(batch['ID'])
        output['pred'].append(y_pred)
    output['pred'] = np.concatenate(output['pred'], 0)  
    output['pred'] = list(output['pred'])  
    output[eff_type] = ['Yes' if x >= cutoff else 'No' for x in output['pred']]
    import pandas as pd
    output = pd.DataFrame(output)
    if output_file:
        try:
            output.to_csv(output_file)
            print(f'Predictions saved as {output_file}')
        except:
            print(f'Predictions failed to save as {output_file}')
            import uuid
            tmp_name = str(uuid.uuid4())+'_clef'
            tag = os.path.split(input_file)[-1]
            output_file = os.path.join(os.path.dirname(input_file), f"./{tag}_{eff_type}_prediction.csv") 
            output.to_csv(output_file)
            print(f'Predictions saved as {output_file}')    
    if Return:
        return output

def generate_protein_representation(input_file,
                    output_file,
                    model_params_path = None,
                    tmp_dir = "./tmp",
                    embedding_generator = fasta_to_EsmRep,
                    esm_config = None,
                    remove_tmp = True,
                    mode = 'clef',
                    maxlength = 256    # Hyperparameter determining how many amino acids are used in protein-encoding by PLM
                    ):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
        print(f'Make a temp directory:{tmp_dir}')
    if is_fasta_file(input_file):
        print(f"Transform representation from fasta file {input_file}")
        import uuid
        tmp_file = str(uuid.uuid4())+'_tmp_esm'
        tmp_file = os.path.join(tmp_dir, tmp_file)
        esm_config = esm_config if isinstance(esm_config, dict) else {'Final_pool':False, 'maxlen':maxlength}
        esm_config['input_fasta'] = input_file
        esm_config['output_file'] = tmp_file
        esm_config['Return'] = False
        if 'pretrained_model_params' not in esm_config:
            esm_config['pretrained_model_params'] = os.path.join(find_root_path(), "./pretrained_model/esm2_t33_650M_UR50D.pt")
        try:
            embedding_generator(**esm_config)
        except:
            print("Failed to transform fasta into ESM embeddings, make sure esm config is correct")
            import shutil
            shutil.rmtree(tmp_dir)
            sys.exit(1)

       
    if mode.lower() == 'clef':
        print(f"Using pre-trained encoder in CLEF to generate protein representations")
        num_hidden = check_hidden_layer_dimensions(load_feature_from_local(tmp_file, silence=True))
        assert num_hidden, "Dimension numbers of the last dimension is not same"
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
        encoder = clef_enc(num_hidden).to(device)
        try:
            encoder.load_state_dict(torch.load(model_params_path, map_location=torch.device('cpu')), strict=False)
            print(f"Successfully load CLEF params from {model_params_path}.")
        except:
            print(f"Failed to load CLEF params from {model_params_path}, make sure it is a valid weights for CLEF")
            import shutil
            shutil.rmtree(tmp_dir)
            sys.exit(1)
        tmp_output= output_file
        loader_config = {'batch_size':64, 'max_num_padding':256}
        config = {
          'input_file':tmp_file,
          'output_file':tmp_output,
          'model':encoder,
          'params_path':None,
          'loader_config':loader_config
        }
        generate_clef_feature(**config)

    elif mode.lower() == 'esm':
        print(f"Direct generate esm representations")
        conf = {
        'input_embeddings_path' : tmp_file,
        'output_file' : output_file,
        }
        generate_ESM_feature(**conf)
        print(f"ESM2 (protein) array saved as {output_file}")
    else:
        print(f"{mode} is not a valid mode tag, please select [clef] or [esm] for protein-reps generation")
        import shutil
        shutil.rmtree(tmp_dir)
        sys.exit(1)
        
    print(f"Done..")
    
    if remove_tmp:
        import shutil 
        try:
            shutil.rmtree(tmp_dir)
            print(f"Remove temp directory: {tmp_dir}.")
        except:
            print(f"Failed to remove temp file in {tmp_dir}.")

def train_clef(input_file_config,
                output_dir,
                model_initial = clef_multimodal,
                tmp_dir = "./tmp",
                embedding_generator = fasta_to_EsmRep,
                esm_config = None,
                train_config = None,
                ):
    log = []
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    assert "seq" in  input_file_config , "For training.sequence representation or fasta file"   
    input_file = input_file_config['seq']
    feat_dim_config = {}
    for key, value in input_file_config.items():
        if key != 'seq':
            feat_dim = check_hidden_layer_dimensions(load_feature_from_local(value, silence=True))
            assert feat_dim, f'Dimension numbers of the last dimension is not same; {value}'
            feat_dim_config[key] = feat_dim
            line = f'{key}--{value}; num_dims:{feat_dim}'
            print(line)
            log.append(f'{line}\n')
    if is_fasta_file(input_file):
        print(f"Transform representation from fasta file {input_file}")
        import uuid
        tmp_file = str(uuid.uuid4())+'_tmp_esm'
        tmp_file = os.path.join(tmp_dir, tmp_file)
        esm_config = esm_config if isinstance(esm_config, dict) else {'Final_pool':False, 'maxlen':256}
        esm_config['input_fasta'] = input_file
        esm_config['output_file'] = tmp_file
        esm_config['Return'] = False
        if 'pretrained_model_params' not in esm_config:
            esm_config['pretrained_model_params'] = os.path.join(find_root_path(), "./pretrained_model/esm2_t33_650M_UR50D.pt")
        try:
            embedding_generator(**esm_config)
        except:
            print("Failed to transform fasta into ESM embeddings, make sure esm config is correct")
            import shutil
            shutil.rmtree(tmp_dir)
            sys.exit(1)
    else:
        print(f"Direct use {input_file} as sequence representation")
        tmp_file = input_file
    import torch.optim as optim
    import random
    import pandas as pd
    from Data_utils import Potein_rep_datasets
    from Module import InfoNCELoss
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    if device == 'cpu':
        print(f'**Note:model will be trained on CPU, it may take very long time')
    num_embeds = check_hidden_layer_dimensions(load_feature_from_local(tmp_file, silence=True))
    assert num_embeds, f'make sure {tmp_file} is a dict with ID:sequence_reps; sequence_reps should be 2D numpy with shape of [protein_length, num_hidden]'
    line = f'Sequnce data--{input_file}; num_dims:{num_embeds}'
    log.append(f'{line}\n')
    model=model_initial(num_embeds, feat_dim_config).to(device)
    data_path_config = {'esm_feature':tmp_file}
    data_path_config.update({key:value for key, value in input_file_config.items() if key != 'seq'})
    Dataset = Potein_rep_datasets(data_path_config)
    if len(Dataset) == 0:
        raise ValueError("Failed to load feature for training")
        import shutil
        shutil.rmtree(tmp_dir)
        sys.exit(1)
    loss_function= InfoNCELoss()
    if train_config:
        for x in ['lr', 'batch_size', 'num_epoch']:
            assert x in  train_config, "'lr', 'batch_size' and 'num_epoch' are needed when not using default train configuration "
        lr = train_config['lr']
        batch_size = train_config['batch_size']
        num_epoch = train_config['num_epoch']
        maxlen = 256 if 'maxlen' not in train_config else train_config['maxlen']
    else:
        lr = 0.00002
        batch_size = 128
        num_epoch = 20
        maxlen = 256
    optimizer=optim.Adam(model.parameters(),lr=lr)
    activation = None
    temp_check_point = output_dir
    import time
    save_by_epoch = False
    #check_list = [10, 15, 20, 25, 30]
    check_list = []
    
    if not os.path.exists(temp_check_point):
        os.mkdir(temp_check_point)
    s = time.time()
    for epoch in range(num_epoch):
        Loss=0
        sum=0 
        for batch in Dataset.Dataloader(batch_size=batch_size,  max_num_padding=maxlen, device=device):
            with torch.no_grad():
                model.eval()
                _, proj, Y = model(batch)
                loss = loss_function(proj, Y)
                assert not np.isnan(loss.item()), 'odd loss value appears'
                Loss += loss.item()
                sum += Y.shape[0]
            model.train()
            _, proj, Y = model(batch)
            loss = loss_function(proj,Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        e = time.time()
        t = e - s
        avg_loss = Loss / sum
        line = f'Epoch: {epoch}; Train loss:{avg_loss}; time:{t} s'
        print(line)
        log.append(f'{line}\n')
        tmp_check_path = os.path.join(temp_check_point, f"./checkpoint_{epoch}.pt")
        if save_by_epoch:
            torch.save(model.state_dict(), tmp_check_path)
        elif epoch == num_epoch - 1 or epoch + 1 in check_list:
            torch.save(model.state_dict(), tmp_check_path)
            line = f'Epoch:{epoch} Checkpoint weights saved--{tmp_check_path}'
            print(line)
            log.append(f'{line}\n')
    e = time.time()
    print(f'Total training time : {e-s} seconds')
    print(f"Done..")
    import shutil 
    try:
        shutil.rmtree(tmp_dir)
    except:
        print(f"Failed to remove temp file in {tmp_dir}.")
    log_path = os.path.join(output_dir, 'log.txt')
    with open(log_path, 'w') as f:
        f.writelines(log)
    print(f"Log text file saved to {log_path}.")  
      
