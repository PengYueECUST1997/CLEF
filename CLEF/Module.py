from torch import nn, einsum
import torch.nn.functional as F
import torch
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64, dropout=0.):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)
        self.to_out = nn.Linear(dim_value * heads, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, mask=None):
        n, h = x.shape[-2], self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        q = q * self.scale
        logits = einsum('b h i d, b h j d -> b h i j', q, k)

        if mask is not None:
            logits.masked_fill(mask, -1e9)

        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn


class TransformerLayer(nn.Module):

    def __init__(self, hid_dim, heads, dropout_rate, att_dropout=0.05):
        super().__init__()
        self.attn = Attention(hid_dim, heads, hid_dim //
                              heads, hid_dim // heads, att_dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Dropout(dropout_rate))
        self.layernorm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):

        residual = x
        x = self.layernorm(x)  # pre-LN
        x, attn = self.attn(x, mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ffn(x)
        x = residual + x

        return x, attn

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.BCELoss()
    
    def forward(self, y_hat, y_label):
        y_label = y_label.to(torch.float32)
        if y_hat.shape[0] != 1 or len(y_hat.shape) != 1:
          y_hat = y_hat.squeeze(-1)
        loss = self.loss_function(y_hat, y_label)
        return loss

class test_dnn(nn.Module):
    def __init__(self, num_embeds = 1280, finial_drop = 0.5, out_dim = 1):
        super().__init__()
        self.dnn = nn.Sequential(nn.Linear(num_embeds, 2 * num_embeds), nn.ReLU(),
                                 nn.Linear(2 * num_embeds, num_embeds))
        self.out = nn.Sequential(nn.Linear(num_embeds, 128), nn.ReLU(),
                                 nn.Linear(128, out_dim))
        self.binaryclass = True if out_dim == 1 else False
        self.Dropout = nn.Dropout(finial_drop)
        self.ln = nn.LayerNorm(num_embeds)
      
    def forward(self, x):
        x = x['feature']
        x = self.ln(self.dnn(x))
        x = self.out(self.Dropout(x))    
        if self.binaryclass:
            x = torch.sigmoid(x).squeeze(-1)  
        
            
        return x  


def sequence_mask(X, valid_lens):
    mask = torch.zeros((X.shape[0], X.shape[1]), dtype = torch.bool).to(X.device)                  
    expanded_valid_lens = valid_lens.view(-1, 1).expand(X.shape[0], X.shape[1])    
    src_key_padding_mask = mask.masked_fill(torch.arange(X.shape[1]).to(X.device).view(1, -1).expand(X.shape[0], X.shape[1]) >= expanded_valid_lens, True)
    return src_key_padding_mask


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds):
        
        image_embeds = F.normalize(image_embeds, dim=-1, p=2)
        text_embeds = F.normalize(text_embeds, dim=-1, p=2)

        similarity_matrix = F.cosine_similarity(image_embeds.unsqueeze(1), text_embeds.unsqueeze(0), dim=-1) / self.temperature

        positives = torch.diag(similarity_matrix)
 
        contrastive_loss = (torch.logsumexp(similarity_matrix - self.margin, dim=-1) - positives).mean()

        return contrastive_loss
