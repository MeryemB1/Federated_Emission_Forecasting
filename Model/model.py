import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TemporalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, F)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # (T,1,F)

    def forward(self, x):
        # x: (B, T, N, F)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # (T,1,F) -> (T,1,1,F)

class GatedGroupedTemporalConv(nn.Module):
    def __init__(self, hidden_dim, kernel_size=6, num_nodes=189):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(kernel_size, 1),
            groups=1
        )
        self.gate = nn.Conv2d(
            in_channels=hidden_dim ,
            out_channels=hidden_dim ,
            kernel_size=(kernel_size, 1),
            groups=1
        )
      #B, T, N, Fe
    def forward(self, x):
        B, T, N, H = x.shape
        x = x.permute(0,3,1,2)  # (B, H, T, N)
        x = F.pad(x, (0, 0, self.kernel_size - 1, 0))
        out = torch.tanh(self.conv(x)) * torch.sigmoid(self.gate(x))
        #.view(B, N, H, T).permute(0, 2, 3, 1)  # (B, N, T, H)

        out = out.permute(0,2,3,1)

        return out

class AttentionTransitionMatrix(nn.Module):
    def __init__(self, in_features, sequence_len, hidden_dim):
        super().__init__()
        self.att_q = nn.Linear(in_features * sequence_len, hidden_dim)
        self.att_k = nn.Linear(in_features * sequence_len, hidden_dim)

    def forward(self, X_seq, adj):
        # X_seq: (B, T, N, F)
        B, T, N, Fe = X_seq.shape
        X_temporal = X_seq.permute(0, 2, 1, 3).reshape(B, N, T * Fe)
        Q = self.att_q(X_temporal)
        K = self.att_k(X_temporal)

        d_k = Q.size(-1)
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        adj_with_loops = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
        # Optional: If binary, clamp to 0/1
        adj_with_loops = (adj_with_loops > 0).to(adj.dtype)
        # 2. Expand to batch size B
        adj_batch = adj_with_loops.unsqueeze(0).expand(B, N, N)
        # 3. Use in masking
        scores = raw_scores.masked_fill(adj_batch == 0, float('-inf'))
        P = F.softmax(scores, dim=-1)
        return P  # (B, N, N)

class BidirectionalDiffusion(nn.Module):
    def __init__(self, hidden_dim, k_hop):
        super().__init__()
        self.k_hop = k_hop
        self.hop_alpha_f = nn.Parameter(torch.zeros(k_hop))
        self.hop_alpha_b = nn.Parameter(torch.zeros(k_hop))
        self.output_proj_f = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj_b = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X_seq, P):
        # X_seq: (B, T, N, H), P: (B, N, N)
        B, T, N, H = X_seq.shape
        f_sum = X_seq.clone()
        b_sum = X_seq.clone()
        v_f = X_seq
        v_b = X_seq
        P = P.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
        for k in range(1, self.k_hop + 1):
            v_b = torch.bmm(P, v_b.reshape(B*T, N, H)).reshape(B, T, N, H)
            b_sum += self.hop_alpha_b[k - 1] * v_b

            v_f = torch.bmm(P, v_f.reshape(B*T, N, H)).reshape(B, T, N, H)
            f_sum += self.hop_alpha_f[k - 1] * v_f

        Fwd = torch.tanh(self.output_proj_f(f_sum))
        Bwd = torch.tanh(self.output_proj_b(b_sum))
        return Fwd , Bwd

class SpatioTemporalAttentionDiffusionNet(nn.Module):
    def __init__(self, in_features, hidden_dim, forecast_len, sequence_len, num_nodes, prejection, k_hop=3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.forecast_len = forecast_len
        self.sequence_len = sequence_len
        self.num_nodes = num_nodes
        self.prejection = prejection
        #self.missing_emb = nn.Parameter(torch.randn(1, 1, num_nodes, in_features))
        self.temporal_enc = TemporalEncoding(self.in_features, max_len=sequence_len)
        self.time_emb = nn.Parameter(torch.randn(sequence_len, 1, in_features))  # (T,1,F)
        self.proj = nn.Linear(in_features + 1, hidden_dim)
        self.temporal1 = GatedGroupedTemporalConv(hidden_dim, kernel_size=3, num_nodes=num_nodes)
        self.temporal_f = GatedGroupedTemporalConv(hidden_dim, kernel_size=3, num_nodes=num_nodes)
        self.temporal_b = GatedGroupedTemporalConv(hidden_dim, kernel_size=3, num_nodes=num_nodes)
        self.last_P = None
        self.transition = AttentionTransitionMatrix(hidden_dim, sequence_len, hidden_dim)
        self.diffusion = BidirectionalDiffusion(hidden_dim, k_hop)
        self.proj_impute = nn.Linear(in_features, hidden_dim)


        self.decoder = nn.Sequential(
            nn.Linear(sequence_len * hidden_dim * 2 , prejection),
            nn.ReLU(),
            nn.Linear(prejection, sequence_len * in_features),
            nn.ReLU(),
        )

        self.decoder_for = nn.Sequential(
            nn.Linear(sequence_len * hidden_dim , prejection),
            nn.ReLU(),
            nn.Linear(prejection, forecast_len * in_features),
            nn.ReLU(),
        )


    def forward(self, X, adj , mask):
        B, T, N, Fe = X.shape

        if mask.shape[-1] != 1:
            mask_channel = mask[..., :1]  # take one channel
        else:
            mask_channel = mask

        # Concatenate mask as extra feature
        X_aug = torch.cat([X, mask_channel.float()], dim=-1)  # (B, T, N, Fe+1)

        # Project to hidden dimension
        x = self.proj(X_aug)                    # (B, T, N, H)
        x = self.temporal1(x)  # (B, T, N, H)
        #x = x.permute(0, 2, 3, 1)  # (B, T, N, H)
        P = self.transition(x, adj)           # (B, N, N)
        self.last_P = P.detach().cpu()
        Fwd,Bwd = self.diffusion(x, P)       # (B, T, N, H)
        Fwd_out = self.temporal_f(Fwd)  # (B, T, N, F)
        Bwd_out = self.temporal_b(Bwd)
        Fwd_out = Fwd_out.permute(0,2,1,3).reshape(B, N, T * self.hidden_dim)
        Bwd_out = Bwd_out.permute(0,2,1,3).reshape(B, N, T * self.hidden_dim)
        x = torch.cat([Fwd_out,Bwd_out], dim=-1)
        imputed = self.decoder(x).view(B, N, self.sequence_len, self.in_features)
        imputed_seq = imputed.permute(0, 2, 1, 3) #(B, T, N, F)
        imputed_hidden = self.proj_impute(imputed_seq) #(B, T, N, H)
        imputed_hidden= imputed_hidden.permute(0, 2, 1, 3).reshape(B, N, T * self.hidden_dim)
        # Collapse last two dims: (B, N)
        mask_reduced = mask.any(dim=1).any(dim=-1).float()   # shape [B, N]
        mask_expanded = mask_reduced.unsqueeze(-1)           # shape [B, N, 1]
        # Broadcast to hidden size
        mask_expanded = mask_expanded.expand(-1, -1, Fwd_out.size(-1))  # [B, N, 128]
        mixed_hidden = mask_expanded * Fwd_out + (1 - mask_expanded) * imputed_hidden
        forcast = self.decoder_for(mixed_hidden).view(B, N, self.forecast_len, self.in_features)
        # out here will represent the imputed values
        return imputed.permute(0, 2, 1, 3), forcast.permute(0, 2, 1, 3)   # (B, sequence_len, N, F) (B, forecast_len, N, F)
        