
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
from CLEF import clef
from Module import test_dnn



    
    

def predict_from_1D_rep(rep_path, model, params_path,
                        output_file = None,
                        Return = True):
    predictset = Potein_rep_datasets({'feature':rep_path})
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if isinstance(model, torch.nn.Module):
        model.eval()
    if params_path:
      print(f"Loading model weights from {params_path}")
      try:
        loaded_params = torch.load(params_path, map_location=device)
        model.load_state_dict(loaded_params)
        print(f"Load classifier weights successfully")
      except:
        print(f"Failed to load model weights from {params_path}")

    output_dict = {
      'ID':[]
    }
    tmp_scores = []
    for batch in predictset.Dataloader(batch_size=64, shuffle=False, test=True, max_num_padding=None, device=device):
        with torch.no_grad():
            y_hat = model(batch)
        output_dict['ID'].extend(batch['ID'])
        y_hat = y_hat.detach().to('cpu').numpy()
        tmp_scores.append(y_hat)
    tmp_scores = np.concatenate(tmp_scores, 0) 
    if len(tmp_scores.shape) > 1:
        for i in range(tmp_scores.shape[1]):
            tag = f"score_{i}"
            output_dict[tag] = tmp_scores[:,i]
    else:
        output_dict['score'] = tmp_scores
    output_dict = pd.DataFrame(output_dict)
    
    if output_file:
        try:
            output_dict.to_excel(output_file)
            print(f'Predictions saved as {output_file}')
        except:
            print(f'Predictions failed to save as {output_file}')
            import uuid
            tmp_name = str(uuid.uuid4())+'_clef'
            output_file =os.path.join(os.path.dirname(input_file), tmp_name) 
            output_dict.to_excel(output_file)
            print(f'Predictions saved as {output_file}')
    
    if Return:
        return output_dict

def generate_protein_representation(input_file,
                    output_file,
                    model_params_dict = None,
                    tmp_dir = "./tmp",
                    embedding_generator = fasta_to_EsmRep,
                    esm_config = None
                    ):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
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
            fasta_to_EsmRep(**esm_config)
        except:
            print("Failed to transform fasta into ESM embeddings, make sure esm config is correct")

    output_dict = {}
       
    try:
        for tag, params in model_params_dict.items():
            encoder_path, encoder_config = params
            encoder = clef(**encoder_config)
            tmp_output= output_file
            loader_config = {'batch_size':64, 'max_num_padding':esm_config['maxlen']}
            config = {
              'input_file':tmp_file,
              'output_file':tmp_output,
              'model':encoder,
              'params_path':encoder_path,
              'loader_config':loader_config
            }
            generate_clef_feature(**config)

    except:
        print(f"No valid encoder params loaded, direct generate esm representations")
        conf = {
        'input_embeddings_path' : tmp_file,
        'output_file' : output_file,
        }
        generate_ESM_feature(**conf)
        print(f"ESM2 (protein) array saved as {output_file}")
    print(f"Done..")
    import shutil 
    try:
        shutil.rmtree(tmp_dir)
    except:
        print(f"Failed to remove temp file in {tmp_dir}.")

    

def predict_by_clef(input_file,
                    output_file,
                    model_params_dict = None,
                    tmp_dir = "./tmp",
                    embedding_generator = fasta_to_EsmRep,
                    esm_config = None,
                    transform_method = generate_clef_feature, 
                    ):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if is_fasta_file(input_file):
        print(f"Transform representation from fasta file {input_file}")
        import uuid
        tmp_file = str(uuid.uuid4())+'_tmp_esm'
        tmp_file = os.path.join(tmp_dir, tmp_file)
        esm_config = esm_config if isinstance(esm_config, dict) else {'Final_pool':False, 'maxlen':256, 'Return':False}
        esm_config['input_fasta'] = input_file
        esm_config['output_file'] = tmp_file
        esm_config['Return'] = False
        esm_config['pretrained_model_params'] = os.path.join(find_root_path(), "../pretained_model/esm2_t33_650M_UR50D.pt")
        try:
            fasta_to_EsmRep(**esm_config)
        except:
            print("Failed to transform fasta into ESM embeddings, make sure esm config is correct")
    else:
        print(f"Direct load esm representations from {input_file}")
    
    output_dict = {}
    if not isinstance(model_params_dict, dict):
        print(f"No valid encoder params loaded, direct use esm representations for prediction")
        conf = {
        'input_embeddings_path' : tmp_file,
        'output_file' :tmp_file+'pool',
        }
        generate_ESM_feature(**conf)
        rep_path = tmp_file+'pool'
        classifier_path = os.path.join(find_root_path(), "./pretained_model/ESM_clef_T3SE_classifier.pt")         
        classifer = test_dnn(1280)
        config = {
              'rep_path':rep_path,
              'output_file':None,
              'model':classifer,
              'params_path':classifier_path,
              'Return':True
        }        
        output_dict = predict_from_1D_rep(**config)
        
    else:
        for tag, params in model_params_dict.items():
            encoder_path, encoder_config = params['encoder']
            encoder = clef(**encoder_config)
            tmp_output= tmp_file.replace('esm', f'{tag}_clef')
            params_path = "../Benchmark-partB/Bastion3/params/encoder_params"
            loader_config = {'batch_size':64, 'max_num_padding':esm_config['maxlen']}
            config = {
              'input_file':tmp_file,
              'output_file':tmp_output,
              'model':encoder,
              'params_path':encoder_path,
              'loader_config':loader_config
            }
            generate_clef_feature(**config)
            
            rep_path = tmp_output
            classifier_path, classifier_config = params['classifier']           
            classifer = test_dnn(**classifier_config)
            config = {
              'rep_path':rep_path,
              'output_file':None,
              'model':classifer,
              'params_path':classifier_path,
              'Return':True
            }        
            tmp_preds = predict_from_1D_rep(**config)
            if len(output_dict) == 0:
                output_dict = tmp_preds
            else:
                assert min(output_dict['ID'] == tmp_preds['ID'])
                output_dict = pd.merge(output_dict, tmp_preds, on='ID', how='inner')
            output_dict.columns = [x if i <len(output_dict.columns) - 1 else f'score_{tag}' for i,x in enumerate(output_dict.columns)]
    
    output_dict.to_excel(output_file, index = False)
    print(f"Predictions saved at {output_file}")
    import shutil 
    try:
        shutil.rmtree(tmp_dir)
    except:
        print(f"Failed to remove temp file in {tmp_dir}.")

def train_clef(input_file_config,
                output_dir,
                model_initial = clef,
                model_config = None,
                tmp_dir = "./tmp",
                embedding_generator = fasta_to_EsmRep,
                esm_config = None,
                train_config = None 
                ):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    assert "fasta" in  input_file_config and "supp_feat" in input_file_config, "PATH for training .fasta file and feature file are needed"   
    
    
