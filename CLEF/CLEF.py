import torch.nn as nn 
import torch
import torch.nn.functional as F
from Module import TransformerLayer, sequence_mask

class clef(nn.Module):
  
  def __init__(self, num_embeds = 1280, feature_dim = 768, num_hiddens = 128, 
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
      
  def forward(self, batch):
    if 'B_feature' not in batch:
        features_input = None
        X, valid_lens = batch['esm_feature'], batch['valid_lens']
    else:
        X, valid_lens, features_input = batch['esm_feature'], batch['valid_lens'], batch['B_feature']
    
    mask = torch.zeros((X.shape[0], X.shape[1]), dtype = torch.bool).to(X.device)                  
    expanded_valid_lens = valid_lens.view(-1, 1).expand(X.shape[0], X.shape[1])    
    src_key_padding_mask = mask.masked_fill(torch.arange(X.shape[1]).to(X.device).view(1, -1).expand(X.shape[0], X.shape[1]) >= expanded_valid_lens, True)
    for layer in self.layers:
      X, _ = layer(X, mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2))
    
    X = torch.cat([X[i, :valid_lens[i] + 2].mean(0).unsqueeze(0)
                        for i in range(X.size(0))], dim=0)
    proj_X = self.mlp(self.Dropout(X))
    
    
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





class clef_PPI(nn.Module):
  
  def __init__(self, num_embeds = 1280, feature_dim = 256, num_hiddens = 128, 
                      finial_drop = 0.25, A_relu = True, B_relu = True,
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
