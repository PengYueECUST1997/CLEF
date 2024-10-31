import torch.nn as nn 
import torch
import torch.nn.functional as F
from Module import TransformerLayer, sequence_mask

def standardize(feature):
    mean = feature.mean(dim=0, keepdim=True)
    std = feature.std(dim=0, keepdim=True)
    return (feature - mean) / std

class clef_enc(nn.Module): # only contains encoder A

    def __init__(self, num_embeds, num_hiddens=128, finial_drop=0.1, mlp_relu=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(num_embeds, 8, 0.45, 0.05)
                for _ in range(2)
            ]
        )
        self.Dropout = nn.Dropout(finial_drop)
        self.ln = nn.LayerNorm(num_embeds)
        if mlp_relu:
            self.mlp = nn.Sequential(nn.Linear(num_embeds, 2 * num_embeds), nn.ReLU(),
                                     nn.Linear(2 * num_embeds, num_hiddens))
        else:
            self.mlp = nn.Linear(num_embeds, num_hiddens)


    def forward(self, batch, Return_res_rep=False):

        X, valid_lens = batch['esm_feature'], batch['valid_lens']

        src_key_padding_mask = sequence_mask(X, valid_lens)
        for layer in self.layers:
            X, _ = layer(X, mask=src_key_padding_mask.unsqueeze(1).unsqueeze(2))

        if not Return_res_rep:   # whether return embeddings per-residue
            X = torch.cat([X[i, :valid_lens[i] + 2].mean(0).unsqueeze(0)
                           for i in range(X.size(0))], dim=0)
            proj_X = self.mlp(self.Dropout(X))
        else:
            proj_X = torch.cat([X[i, :valid_lens[i]].mean(0).unsqueeze(0)
                                for i in range(X.size(0))], dim=0)
            proj_X = self.mlp(self.Dropout(proj_X))

        return X, proj_X



class clef_multimodal(nn.Module):

    def __init__(self, num_embeds, feat_dim_config, num_hiddens=128,
                 finial_drop=0.1, mlp_relu=True, feat_mlp_relu=True,
                 feature_norm=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(num_embeds, 8, 0.45, 0.05)
                for _ in range(2)
            ]
        )
        self.Dropout = nn.Dropout(finial_drop)
        self.ln = nn.LayerNorm(num_embeds)
        if mlp_relu:
            self.mlp = nn.Sequential(nn.Linear(num_embeds, 2 * num_embeds), nn.ReLU(),
                                     nn.Linear(2 * num_embeds, num_hiddens))
        else:
            self.mlp = nn.Linear(num_embeds, num_hiddens)

        self.modal_indeces = {key:i for i, key in enumerate(feat_dim_config)}

        feat_dim = sum([dim for dim in feat_dim_config.values()])
        self.feat_encoder = nn.Sequential(nn.Linear(feat_dim, 1024, bias=False), nn.ReLU(),
                              nn.Linear(1024, num_hiddens))




        self.feature_norm = feature_norm
        if self.feature_norm:
            self.ln_f = nn.LayerNorm(num_hiddens)

    def forward(self, batch, Return_res_rep=False):

        has_modal = min([x in batch for x in self.modal_indeces.keys()])
        X, valid_lens = batch['esm_feature'], batch['valid_lens']

        src_key_padding_mask = sequence_mask(X, valid_lens)
        for layer in self.layers:
            X, _ = layer(X, mask=src_key_padding_mask.unsqueeze(1).unsqueeze(2))

        if not Return_res_rep:   # If return embeddings per-residue
            X = torch.cat([X[i, :valid_lens[i] + 2].mean(0).unsqueeze(0)
                           for i in range(X.size(0))], dim=0)
            proj_X = self.mlp(self.Dropout(X))
        else:
            proj_X = torch.cat([X[i, :valid_lens[i]].mean(0).unsqueeze(0)
                                for i in range(X.size(0))], dim=0)
            proj_X = self.mlp(self.Dropout(proj_X))

        if has_modal:
            cross_features = []
            for modal, i in self.modal_indeces.items():
                if modal in batch:
                    norm_feat = standardize(batch[modal]) if len(self.modal_indeces.items())>1 else batch[modal]
                    cross_features.append(norm_feat)
            cross_features = torch.cat(cross_features, -1)
            proj_features = self.feat_encoder(cross_features)
            if self.feature_norm:
                proj_features = self.ln_f(proj_features)

            return X, proj_X, proj_features
        else:
            return X, proj_X


class clef_multimodal_mlp(nn.Module):

    def __init__(self, num_embeds, feat_dim_config, num_hiddens=128,
                 finial_drop=0.45, mlp_relu=True, feat_mlp_relu=True,
                 feature_norm=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(num_embeds, 2*num_embeds), nn.ReLU(), nn.Linear(2*num_embeds, num_embeds))
                for _ in range(1)
            ]
        )
        self.Dropout = nn.Dropout(finial_drop)
        self.ln = nn.LayerNorm(num_embeds)
        if mlp_relu:
            self.mlp = nn.Sequential(nn.Linear(num_embeds, 2 * num_embeds), nn.ReLU(),
                                     nn.Linear(2 * num_embeds, num_hiddens))
        else:
            self.mlp = nn.Linear(num_embeds, num_hiddens)

        self.modal_indeces = {key:i for i, key in enumerate(feat_dim_config)}

        feat_dim = sum([dim for dim in feat_dim_config.values()])
        self.feat_encoder = nn.Sequential(nn.Linear(feat_dim, 1024, bias=False), nn.ReLU(),
                              nn.Linear(1024, num_hiddens))


        self.feature_norm = feature_norm
        if self.feature_norm:
            self.ln_f = nn.LayerNorm(num_hiddens)

    def forward(self, batch, Return_res_rep=False):

        has_modal = bool(set(self.modal_indeces) & set(batch))
        X, valid_lens = batch['esm_feature'], batch['valid_lens']
        X = torch.cat([X[i, :valid_lens[i] + 2].mean(0).unsqueeze(0)
                           for i in range(X.size(0))], dim=0)
        for layer in self.layers:
            X = layer(X)
        proj_X = self.mlp(self.Dropout(X))

        if has_modal:
            cross_features = []
            for modal, i in self.modal_indeces.items():
                if modal in batch:
                    norm_feat = standardize(batch[modal])
                    cross_features.append(norm_feat)
            cross_features = standardize(torch.cat(cross_features, -1))
            proj_features = self.feat_encoder(cross_features)
            if self.feature_norm:
                proj_features = self.ln_f(proj_features)

            return X, proj_X, proj_features
        else:
            return X, proj_X        


class clef_PPI(nn.Module):
  
  def __init__(self, num_embeds = 1280, feature_dim = 256, num_hiddens = 128, 
                      finial_drop = 0.45, A_relu = True, B_relu = True,
                     feature_norm = True):
      super().__init__()
      self.encoderA = nn.Sequential(nn.Linear(num_embeds, 2 * num_embeds), nn.ReLU(),
                                   nn.Linear(2 * num_embeds, num_embeds))
      self.lnA = nn.LayerNorm(num_embeds)  
      self.Drop = nn.Dropout(finial_drop)
      self.encoderB = nn.Sequential(nn.Linear(feature_dim, 2 * feature_dim), nn.ReLU(),
                                   nn.Linear(2 * feature_dim, feature_dim))
      
      
      if A_relu:
          self.mlp = nn.Sequential(nn.Linear(num_embeds,  num_embeds), nn.ReLU(),
                                 nn.Linear(num_embeds, num_hiddens))
      else:
          self.mlp = nn.Linear(num_embeds, num_hiddens)
      
      if B_relu:
          self.mlp_feat = nn.Sequential(nn.Linear(feature_dim, feature_dim, bias = False), nn.ReLU(),
                                 nn.Linear(feature_dim, num_hiddens))
      else:
          self.mlp_feat = nn.Linear(feature_dim, num_hiddens, bias = False)
      
      self.ln_f = nn.LayerNorm(num_hiddens) if feature_norm else None
      
      
  def forward(self, batch):
      
      if 'B_feature' not in batch:
          features_input = None
          X = batch['esm_feature']
      else:
          X, features_input = batch['esm_feature'], batch['B_feature']
  
      X = self.lnA(self.encoderA(X))
      proj_X = self.mlp(self.Drop(X))
      
      if features_input is not None:
          features_input = self.encoderB(features_input)
          proj_feat = self.mlp_feat(features_input)
          proj_feat = self.ln_f(proj_feat) if self.ln_f else proj_feat
      
          return X, proj_X, proj_feat
      else:
          return X, proj_X

    
        
class clef(nn.Module): # old version
  
  def __init__(self, num_embeds = 1280, feature_dim = 2437, num_hiddens = 128, 
                      finial_drop = 0.1,mlp_relu = True, feat_mlp_relu = True,
                     feature_norm = True):
    super().__init__()
    self.layers = nn.ModuleList(
            [
                TransformerLayer(num_embeds, 8, 0.45, 0.05)
                for _ in range(2)
            ]
        )
    self.out = nn.Linear(num_embeds, feature_dim, bias = False)
    self.Dropout = nn.Dropout(finial_drop)
    self.ln = nn.LayerNorm(num_embeds)
    if mlp_relu:
      self.mlp = nn.Sequential(nn.Linear(num_embeds, 2 * num_embeds), nn.ReLU(),
                               nn.Linear(2 * num_embeds, num_hiddens))
    else:
      self.mlp = nn.Linear(num_embeds, num_hiddens)
    if feat_mlp_relu:
      self.mlp_feat = nn.Sequential(nn.Linear(feature_dim, 1280, bias = False), nn.ReLU(),
                               nn.Linear(1280, num_hiddens))
    else:
      self.mlp_feat = nn.Linear(feature_dim, num_hiddens, bias = False)
    self.feature_norm = feature_norm
    if self.feature_norm:
      self.ln_f = nn.LayerNorm(num_hiddens)
      
  def forward(self, batch, Return_res_rep = False):
    if 'B_feature' not in batch:
        features_input = None
        X, valid_lens = batch['esm_feature'], batch['valid_lens']
    else:
        X, valid_lens, features_input = batch['esm_feature'], batch['valid_lens'], batch['B_feature']
        
    src_key_padding_mask = sequence_mask(X, valid_lens)
    for layer in self.layers:
      X, _ = layer(X, mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2))
    
    if not Return_res_rep:
        X = torch.cat([X[i, :valid_lens[i] + 2].mean(0).unsqueeze(0)
                            for i in range(X.size(0))], dim=0)
        proj_X = self.mlp(self.Dropout(X))
    else:
        proj_X = torch.cat([X[i, :valid_lens[i] + 2].mean(0).unsqueeze(0)
                            for i in range(X.size(0))], dim=0)
        proj_X = self.mlp(self.Dropout(proj_X))
    
    
    if features_input is not None:
        features_input = features_input.clamp(-10, 10)
        features_input[torch.isnan(features_input)] = 0

        if len(features_input.shape) == 1:
          features_input = features_input.unsqueeze(-1)
        
        proj_feat = self.mlp_feat(features_input)
        if features_input.shape[-1] != 1 and self.feature_norm: 
          proj_feat = self.ln_f(proj_feat)
    
        return X, proj_X, proj_feat
    else:
        return X, proj_X
