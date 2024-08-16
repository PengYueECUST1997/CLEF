import os
import numpy as np
import pickle
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
'''
import torch
import esm
'''


def fasta_to_OneHot(input_file, output_file = None, maxlen = 512, Return = True, FS_alphabet = False):
  '''
  input_file : input local fasta file path 
  output_file : output encoded file path 
  '''
  if FS_alphabet:
      aa_dict = {amino_acid: i for i, amino_acid in enumerate("DPLAVQSNCEHFRGWKYTIM")}
  else:
      aa_dict = {amino_acid: i for i, amino_acid in enumerate("ACDEFGHIKLMNPQRSTVWY")}
  output_dict = {}
  for record in SeqIO.parse(open(input_file), 'fasta'):
    if maxlen:
      record.seq = record.seq[:maxlen]
    seq_index = np.array([aa_dict[aa] if aa in aa_dict else 20 for aa in record.seq ]).astype(np.int32)
    seq_feature = np.eye(21)[seq_index].astype(np.int32)
    output_dict[record.id] = seq_feature
  if output_file:
      try:
          with open(output_file, 'wb') as f:
            pickle.dump(output_dict, f)
          print(f'One-hot array saved as {output_file}')
      except:
          print(f'One-hot array failed to save as {output_file}')
          import uuid
          tmp_name = str(uuid.uuid4())
          output_file =os.path.join(os.path.dirname(input_file), tmp_name) 
          with open(output_file, 'wb') as f:
            pickle.dump(output_dict, f)
          print(f'Temp One-hot array saved as {output_file}')
  if Return:
      return output_dict


def fasta_to_EsmRep(input_fasta, output_file = None, 
                      pretrained_model_params = '../pretained_model/esm2_t33_650M_UR50D.pt',
                      maxlen = 256,
                      Return = True, 
                      Final_pool = False):
  '''
  input_file : input local fasta file path 
  output_file : output encoded file path 
  '''
  import torch
  import esm
  aa_dict = {amino_acid: i for i, amino_acid in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  try:
      input_embedding_net, alphabet = esm.pretrained.load_model_and_alphabet_local(pretrained_model_params)
  except:
      print(f"Skip loading local pre-trained ESM2 model from {pretrained_model_params}.\nTry to download ESM2-650M from fair-esm")
      input_embedding_net, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
  batch_converter = alphabet.get_batch_converter()
  input_embedding_net = input_embedding_net.to(device)
  input_embedding_net.eval()
  output_dict = {}
  real_maxlen = max(1, maxlen - 2)
  num_layer = len(input_embedding_net.layers)
  for record in SeqIO.parse(open(input_fasta), 'fasta'):
    sequence = str(record.seq[: real_maxlen])  
    sequence = "".join([x if x in aa_dict else 'X' for x in sequence])
    data = [
    ("protein1", sequence),
      ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
      results = input_embedding_net(batch_tokens, repr_layers=[num_layer], return_contacts=True)
    token_representations = results["representations"][num_layer]
    embedding = np.array(token_representations.squeeze(0).detach().to('cpu')).astype(np.float16)
    embedding = embedding[:real_maxlen + 2, ]
    embedding = embedding.mean(0) if Final_pool else embedding
    output_dict[record.id] = embedding
  if output_file:
      try:
          with open(output_file, 'wb') as f:
            pickle.dump(output_dict, f)
          print(f'ESM2 array saved as {output_file}')
      except:
          print(f'ESM2 array failed to save as {output_file}')
          import uuid
          tmp_name = str(uuid.uuid4())+'_esm'
          output_file =os.path.join(os.path.dirname(input_file), tmp_name) 
          with open(output_file, 'wb') as f:
            pickle.dump(output_dict, f)
          print(f'Temp ESM2 array saved as {output_file}')
  if Return:
      return output_dict


def is_fasta_file(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            return first_line.startswith(">")
    except Exception:
        return False


def extract_sequence_label(text):
    import re
    pattern = r'(sp|tr|gb|ref|emb|pdb|dbj)\|([^|]+)\|'
    match = re.search(pattern, text)
    if match:
        return match.group(2)
    return None

def generate_clef_feature(input_file, 
                          output_file,
                          model,
                          params_path = None,
                          loader_config = {'batch_size':64, 'max_num_padding':256},
                          esm_config = {'Final_pool':False, 'maxlen':256, 'Return':False},
                          MLP_proj = False,
                          res_rep = False,
                          Return = False):
    from Data_utils import Potein_rep_datasets
    import torch
    if is_fasta_file(input_file):
        print(f"Transform representation from fasta file {input_file}")
        import uuid
        tmp_file = str(uuid.uuid4())+'_tmp'
        tmp_file = os.path.join(os.path.dirname(input_file), tmp_file)
        esm_config = esm_config if isinstance(esm_config, dict) else {'Final_pool':False, 'maxlen':256, 'Return':False}
        esm_config['input_fasta'] = input_file
        esm_config['output_file'] = tmp_file
        esm_config['Return'] = False
        try:
            fasta_to_EsmRep(**esm_config)
        except:
            print("Failed to transform fasta into ESM embeddings, make sure esm config is correct")
        tmpset = Potein_rep_datasets({'esm_feature':tmp_file})
        try:
            os.remove(tmp_file)
            print("Tmp esm file {tmp_file} removed.")
        except:
            pass
    else:
        print(f"Direct load esm representations from {input_file}")
        tmpset = Potein_rep_datasets({'esm_feature':input_file})
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if isinstance(model, torch.nn.Module):
        model.eval()
    if params_path:
      print(f"Try to load model weights from {params_path}")
      try:
        loaded_params = torch.load(params_path, map_location=device)
        model.load_state_dict(loaded_params)
        print(f"Load model weights successfully")
      except:
        print(f"Failed to load model weights from {params_path}")
    
    loader_config = loader_config if isinstance(loader_config, dict) else {'batch_size':64, 'max_num_padding':256}
    loader_config['shuffle'] = False
    loader_config['device'] = device
    IDs = []
    features = []
    for batch in tmpset.Dataloader(**loader_config):
        with torch.no_grad():
            feat, proj_feat = model(batch, Return_res_rep = res_rep)
        feature = proj_feat if MLP_proj else feat
        feature_list = [feature[i,:].detach().to('cpu').numpy() for i in range(feature.shape[0])]
        IDs.extend(batch['ID'])
        features.extend(feature_list)
    output_dict = {ID:feat.astype(np.float16) for ID, feat in zip(IDs, features)}
    
    if output_file:
        try:
            with open(output_file, 'wb') as f:
              pickle.dump(output_dict, f)
            print(f'CLEF array saved as {output_file}')
        except:
            print(f'CLEF array failed to save as {output_file}')
            import uuid
            tmp_name = str(uuid.uuid4())+'_clef'
            output_file =os.path.join(os.path.dirname(input_file), tmp_name) 
            with open(output_file, 'wb') as f:
              pickle.dump(output_dict, f)
            print(f'Temp CLEF array saved as {output_file}')
    if Return:
        return output_dict

    

def generate_msa_transformer_feat(input_alignments_dir,  
                                   output_file, 
                                   maxlen = 1024, maxmsa = 8, 
                                   suffix = 'fasta',
                                   clust_pool = False,
                                   res_pool = False,
                                   remapping_fasta = None,
                                   Return = True):
  import torch
  import esm
  aa_dict = {amino_acid: i for i, amino_acid in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
  model, alphabet = esm.pretrained.load_model_and_alphabet_local('../pretained_model/esm_msa1b_t12_100M_UR50S.pt')
  model = model.to('cuda:0').eval()
  batch_converter = alphabet.get_batch_converter()
  output_dict = {}
  for alignment in os.listdir(input_alignments_dir)  :
    file = os.path.join(input_alignments_dir, alignment)
    if not is_fasta_file:
        continue
    data = []
    ID = alignment.split(f'.{suffix}')[0]
    for record in SeqIO.parse(open(file), 'fasta'):
      if len(data) > maxmsa - 1:
        break
      sequence = str(record.seq)[:maxlen - 1]
      sequence = ''.join([x if x  in aa_dict else '-' for x in sequence])
      data.append((record.id, sequence))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to('cuda:0')
    with torch.no_grad():
      results = model(batch_tokens, repr_layers=[12], return_contacts=True)
    token_representations = results["representations"][12]
    embedding = np.array(token_representations.squeeze(0).detach().to('cpu')).astype(np.float16)
    if res_pool:
        embedding = embedding.mean(1)
    if clust_pool:
        embedding = embedding.mean(0)

    output_dict[ID] = embedding
    
  
  if remapping_fasta:
      try:
          mapped_output = {}
          for record in SeqIO.parse(open(remapping_fasta), "fasta"):
              if  '|' in record.id:
                 ID = extract_sequence_label(record.id) if extract_sequence_label(record.id) else record.id.split('|')[0]
              else:
                 ID = record.id
              if ID not in output_dict:
                 print(f'{record.id} failed to mapped the msa array id with {ID}')
                 continue
              mapped_output[record.id] = output_dict[ID]
              #print(mapped_output[record.id][0])
          print(f"{len(mapped_output)} IDs mapped with {remapping_fasta}")
          output_dict = mapped_output
      except:
          print(f"Failed to mapped with {remapping_fasta}")
  
  if output_file:
        try:
            with open(output_file, 'wb') as f:
              pickle.dump(output_dict, f)
            print(f'MSA array saved as {output_file}')
        except:
            print(f'MSA array failed to save as {output_file}')
            import uuid
            tmp_name = str(uuid.uuid4())+'_clef'
            output_file =os.path.join(os.path.dirname(input_file), tmp_name) 
            with open(output_file, 'wb') as f:
              pickle.dump(output_dict, f)
            print(f'Temp MSA array saved as {output_file}')
  if Return:
        return output_dict


def generate_biobert_embedding(input_file, output_file,
                                maxlen = 128,
                                final_pool = True,
                                remapping_fasta = None,
                                Return = True):
    from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
    import torch
    with open(input_file, "rb") as f:
      input_anot = pickle.load(f)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)
    model.eval()
    output_dict = {}
    for key, value in  input_anot.items():
        if ';' in value:
            temp = " ".join([value.split("; ")[0], ";", value.split(";")[1]])
        else:
            temp = value
        temp = temp.replace("\n", '.')
        tokens = temp
        encoded_input = tokenizer(tokens, return_tensors='pt', padding = False).to(model.device)
        output = model(**encoded_input)
        embedding = np.array(output.last_hidden_state.detach().to('cpu').squeeze(0)).astype(np.float16)
        embedding = embedding[:maxlen]
        if final_pool:
          embedding = embedding.mean(0)
        output_dict[key] = embedding    
    
    if remapping_fasta:
        try:
            mapped_output = {}
            for record in SeqIO.parse(open(remapping_fasta), "fasta"):
                if  '|' in record.id:
                   ID = extract_sequence_label(record.id) if extract_sequence_label(record.id) else record.id.split('|')[0]
                else:
                   ID = record.id
                if ID not in output_dict:
                   print(f'{record.id} failed to mapped the Biobert array id with {ID}')
                   continue
                mapped_output[record.id] = output_dict[ID]
            print(f"{len(mapped_output)} IDs mapped with {remapping_fasta}")
            output_dict = mapped_output
        except:
            print(f"Failed to mapped with {remapping_fasta}")
    
    if output_file:
        try:
            with open(output_file, 'wb') as f:
              pickle.dump(output_dict, f)
            print(f'Biobert array saved as {output_file}')
        except:
            print(f'Biobert array failed to save as {output_file}')
            import uuid
            tmp_name = str(uuid.uuid4())+'_clef'
            output_file =os.path.join(os.path.dirname(input_file), tmp_name) 
            with open(output_file, 'wb') as f:
              pickle.dump(output_dict, f)
            print(f'Temp Biobert array saved as {output_file}')
     
    if Return:
        return output_dict
 

