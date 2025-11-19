import torch.nn as nn
import torch
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class simpleNet(nn.Module):
    def __init__(self, input_dim):
        super(simpleNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Expert(nn.Module):
    def __init__(self, input_dim):
        super(Expert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class ExpertHead(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(ExpertHead, self).__init__()
        self.expertHead = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])

    def forward(self, x_chunk):
        expert_outputs = [expert(x_chunk[i]) for i, expert in enumerate(self.expertHead)]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        # breakpoint()
        # expert_outputs = expert_outputs * gate_head.squeeze(1).unsqueeze(2)
        return expert_outputs


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None):
        super().__init__()
        # self.linear_re = nn.Sequential(nn.Linear(7 * dim, dim), QuickGELU(), nn.BatchNorm1d(dim))
        self.linear_re = nn.Sequential(nn.Linear(3 * dim, dim), QuickGELU(), nn.BatchNorm1d(dim))
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, y):
        B, N, C = y.shape
        x = self.linear_re(x)
        q = self.q_(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        gates = attn.softmax(dim=-1)
        return gates

    def forward_(self, x):
        x = self.direct_gate(x)
        return x.unsqueeze(1)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, head):
        super(GatingNetwork, self).__init__()
        self.gate = CrossAttention(input_dim, head)

    def forward(self, x, y):
        gates = self.gate(x, y)
        return gates


class MoM(nn.Module):
    def __init__(self, input_dim, num_experts, head):
        super(MoM, self).__init__()
        self.head_dim = input_dim // head
        self.head = head
        self.experts = nn.ModuleList(
            [ExpertHead(self.head_dim, num_experts) for _ in range(head)])
        self.gating_network = GatingNetwork(input_dim, head)

    def forward(self, x1, x2, x3):
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
            x3 = x3.unsqueeze(0)
        x1_chunk = torch.chunk(x1, self.head, dim=-1)
        x2_chunk = torch.chunk(x2, self.head, dim=-1)
        x3_chunk = torch.chunk(x3, self.head, dim=-1)
        # head_input = [[x1_chunk[i], x2_chunk[i], x3_chunk[i], x4_chunk[i], x5_chunk[i], x6_chunk[i], x7_chunk[i]] for i
        #               in range(self.head)]
        head_input = [[x1_chunk[i], x2_chunk[i], x3_chunk[i]] for i
                      in range(self.head)]
        # query = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=-1)
        # key = torch.stack([x1, x2, x3, x4, x5, x6, x7], dim=1)
        # query = torch.cat([x1, x2, x3], dim=-1)
        # key = torch.stack([x1, x2, x3], dim=1)
        # gate_heads = self.gating_network(query, key)
        expert_outputs = [expert(head_input[i]) for i, expert in enumerate(self.experts)]
        outputs = torch.cat(expert_outputs, dim=-1).flatten(start_dim=1, end_dim=-1)
        loss = 0
        if self.training:
            return outputs, loss
        return outputs


def KL_regular(mu_1, logvar_1, mu_2, logvar_2,  mu_3, logvar_3):

    kl_loss1 = -(1 + logvar_1 - mu_1.pow(2) - logvar_1.exp()) / 2  
    kl_loss1 = kl_loss1.sum(dim=1).mean()

    kl_loss2 = -(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp()) / 2  
    kl_loss2 = kl_loss2.sum(dim=1).mean()

    kl_loss3 = -(1 + logvar_3 - mu_3.pow(2) - logvar_3.exp()) / 2  
    kl_loss3 = kl_loss3.sum(dim=1).mean()

    # kl_loss4 = -(1 + logvar_4 - mu_4.pow(2) - logvar_4.exp()) / 2  
    # kl_loss4 = kl_loss4.sum(dim=1).mean()
    # kl_loss = kl_loss1 + kl_loss2 +kl_loss3 +kl_loss4
    kl_loss = kl_loss1 + kl_loss2 + kl_loss3
    # kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2  
    # kl_loss = kl_loss.sum(dim=1).mean()
    # KL_loss = KL_loss1 + KL_loss2 +KL_loss3 + kl_loss
    KL_loss = kl_loss
    return KL_loss


class GraphLearn(nn.Module):
    def __init__(self, input_dim):
        super(GraphLearn, self).__init__()
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = 0   # 阈值
    
    
    def forward(self, x):
        diff = torch.cdist(x, x)  # (B, P, P)
        diff = (diff + self.threshold) * self.t
        adjs = 1 - torch.sigmoid(diff)  # (B, P, P)
        mask = (adjs > self.th).float()  # (B, P, P)
        adjs = adjs * mask
        return adjs
    

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
     
    def forward(self, input, adj):
        support = torch.matmul(input, self.W)  # [batch_size, num_nodes, out_features]
        output = torch.bmm(adj, support)  # [batch_size, num_nodes, out_features]
        self.output = output
        return output
    
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(input_dim, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj)) + x
        x = F.relu(self.gcn2(x, adj)) + x
        return x
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    D = torch.sum(adj, dim=-1)
    d_inv_sqrt = torch.pow(D, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0. 
    D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    adj_normalized = torch.bmm(torch.bmm(D_inv_sqrt, adj), D_inv_sqrt)
    return adj_normalized      




def load_balanced_loss(router_probs, expert_mask):
    num_experts = expert_mask.size(-1)

    density = torch.mean(expert_mask, dim=0)
    density_proxy = torch.mean(router_probs, dim=0)
    loss = torch.mean(density_proxy * density) * (num_experts ** 2)

    return loss

class Gate(nn.Module):
    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x, use_aux_loss=True):
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        loss_gate_scores = gate_scores
        # print(gate_scores)
        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(4, dim=-1) #3/4
        top_k_scores_1, top_k_indices_1 = gate_scores.topk(1, dim=-1) #1/4

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )
        mask_1 = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices_1, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask
        masked_gate_scores_1 = gate_scores * mask_1

        # Denominators
        # denominators = (
        #     masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        # )

        # Norm gate scores to sum to the capacity
        # gate_scores = (masked_gate_scores / denominators) * capacity
        gate_scores = masked_gate_scores
        gate_scores_1 = masked_gate_scores_1

        if use_aux_loss:
            # load = gate_scores.sum(0)  # Sum over all examples
            # importance = gate_scores.sum(1)  # Sum over all experts

            # # Aux loss is mean suqared difference between load and importance
            # loss = ((load - importance) ** 2).mean()
            # breakpoint()
            loss = load_balanced_loss(loss_gate_scores,mask)

            return gate_scores, loss, gate_scores_1

        return gate_scores, None


class FeedForward(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim,output_dim)
        self.logvar = nn.Linear(input_dim,output_dim)

    def reparameterise(self,mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def KL_loss(self,mu,logvar):
        return (-(1+logvar-mu.pow(2)-logvar.exp())/2).sum(dim=1).mean()

    def forward(self,x):
        mu=self.mu(x) + x
        logvar = self.logvar(x)
        kl_loss = self.KL_loss(mu,logvar)
        return mu , kl_loss ,torch.exp(logvar)

class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, dim)
                for _ in range(num_experts)
            ]
        )
        self.gate = Gate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x, y, z):

        gate_scores, loss, gate_scores_1 = self.gate(x, use_aux_loss=self.use_aux_loss)
        
        # breakpoint()
        expert_outputs = []
        loss_kl = []
        Uncertainty =[]

        expert_outputs_y = []
        loss_kl_y = []
        Uncertainty_y =[]

        expert_outputs_z = []
        loss_kl_z = []
        Uncertainty_z =[]



        for expert_output, kl_loss,sigma in [expert(x) for expert in self.experts]:
            expert_outputs.append(expert_output)
            loss_kl.append(kl_loss)
            Uncertainty.append(sigma)

        for expert_output_y, kl_loss_y,sigma_y in [expert(y) for expert in self.experts]:
            expert_outputs_y.append(expert_output_y)
            loss_kl_y.append(kl_loss_y)
            Uncertainty_y.append(sigma_y)

        for expert_output_z, kl_loss_z,sigma_z in [expert(z) for expert in self.experts]:
            expert_outputs_z.append(expert_output_z)
            loss_kl_z.append(kl_loss_z)
            Uncertainty_z.append(sigma_z)
        
        loss_KL=0
        for i in range(self.num_experts):
            loss_KL+=loss_kl[i]
        
        loss = loss +(loss_KL)/self.num_experts
        Uncertainty = torch.stack(Uncertainty).sum(2).permute(1,0)
        # breakpoint()
        loss_u=(Uncertainty * gate_scores).sum(1).mean()
        loss = loss + loss_u
        


        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, output_dim, num_experts)
        
        # Stack and weight outputs
        stacked_expert_outputs_y = torch.stack(
            expert_outputs_y, dim=-1
        )  # (batch_size, output_dim, num_experts)

        # Stack and weight outputs
        stacked_expert_outputs_z = torch.stack(
            expert_outputs_z, dim=-1
        )  # (batch_size, output_dim, num_experts)

        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0
        if torch.isnan(stacked_expert_outputs_y).any():
            stacked_expert_outputs_y[
                torch.isnan(stacked_expert_outputs_y)
            ] = 0
        if torch.isnan(stacked_expert_outputs_z).any():
            stacked_expert_outputs_z[
                torch.isnan(stacked_expert_outputs_z)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )
        
        moe_output_y = torch.sum(
            gate_scores_1.unsqueeze(-2) * stacked_expert_outputs_y, dim=-1
        )

        moe_output_z = torch.sum(
            gate_scores_1.unsqueeze(-2) * stacked_expert_outputs_z, dim=-1
        )
        # breakpoint()

        return moe_output, moe_output_y, moe_output_z, loss

class MoE_block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mult: int = 4,
        dropout: float = 0.1,
        num_experts: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout



        self.ffn1 = MoE(
            dim, dim * mult, dim, num_experts, *args, **kwargs
        )
        
        self.ffn2 = MoE(
            dim, dim * mult, dim, num_experts, *args, **kwargs
        )
        
        self.ffn3 = MoE(
            dim, dim * mult, dim, num_experts, *args, **kwargs
        )
        


    def forward(self, x, y, z):
        loss = 0
        x, y_1, z_1, loss1 = self.ffn1(x, y, z)
        y, x_2, z_2, loss2 = self.ffn2(y, x, z)
        z, x_3, y_3, loss3 = self.ffn3(z, x, y)
        x = x + x_2 + x_3
        y = y + y_1 + y_3
        z = z + z_1 + z_2
        outputs = torch.cat([x, y, z], dim = 1)
        loss = loss1 + loss2 + loss3
        if self.training:
            return outputs, 1e-4*loss
        return outputs

class GeneralFusion(nn.Module):
    def __init__(self, feat_dim, num_experts, head, reg_weight=0.1, dropout=0.1, cfg=None):
        super(GeneralFusion, self).__init__()
        self.reg_weight = reg_weight
        self.feat_dim = feat_dim

        self.GPGR = cfg.MODEL.GPGR
        self.UGMoE = cfg.MODEL.UGMoE

        if self.GPGR:
            self.dropout = dropout
            self.GraphConstructR = GraphLearn(input_dim = self.feat_dim)
            self.GraphConstructT = GraphLearn(input_dim = self.feat_dim)
            self.GraphConstructN = GraphLearn(input_dim = self.feat_dim)
            self.MessagePassingR = GCN(input_dim= self.feat_dim, hidden_dim = self.feat_dim, output_dim= self.feat_dim, dropout=self.dropout)
            self.MessagePassingT = GCN(input_dim= self.feat_dim, hidden_dim = self.feat_dim, output_dim= self.feat_dim, dropout=self.dropout)
            self.MessagePassingN = GCN(input_dim= self.feat_dim, hidden_dim = self.feat_dim, output_dim= self.feat_dim, dropout=self.dropout)
            self.MessagePassingR1 = GCN(input_dim= self.feat_dim, hidden_dim = self.feat_dim, output_dim= self.feat_dim, dropout=self.dropout)
            self.MessagePassingT1 = GCN(input_dim= self.feat_dim, hidden_dim = self.feat_dim, output_dim= self.feat_dim, dropout=self.dropout)
            self.MessagePassingN1 = GCN(input_dim= self.feat_dim, hidden_dim = self.feat_dim, output_dim= self.feat_dim, dropout=self.dropout)
            self.w = nn.Parameter(torch.zeros(3))
            
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.rgb_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim),QuickGELU())
            self.nir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                        nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())
            self.tir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                        nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())
        
        

            
        
        if self.UGMoE:
            self.moe = MoE_block(num_tokens=16, dim=self.feat_dim, heads=8, dim_head=64)


        self.unc_logvar1 =  nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_mu1 = nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_logvar2 =  nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_mu2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_logvar3 =  nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_mu3 = nn.Linear(self.feat_dim, self.feat_dim)
    
        self.unc_logvar1_p =  nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_mu1_p = nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_logvar2_p =  nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_mu2_p = nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_logvar3_p =  nn.Linear(self.feat_dim, self.feat_dim)
        self.unc_mu3_p = nn.Linear(self.feat_dim, self.feat_dim)

    

    def forward_UGMoE(self, RGB_special, NI_special, TI_special):
        if self.training:
            moe_feat, loss_reg = self.moe(RGB_special, NI_special, TI_special)
            return moe_feat,  loss_reg
        else:
            moe_feat = self.moe(RGB_special, NI_special, TI_special)
            return moe_feat
        

    def reparameterise(self,mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        if self.GPGR:
            u_c1 = self.unc_logvar1(RGB_global) 
            s_c1 = self.unc_mu1(RGB_global)
            u_c2 = self.unc_logvar2(NI_global) 
            s_c2 = self.unc_mu2(NI_global)
            u_c3 = self.unc_logvar3(TI_global) 
            s_c3 = self.unc_mu3(TI_global)
            un_loss = KL_regular(u_c1, s_c1, u_c2, s_c2, u_c3, s_c3)

            u_p1 = self.unc_mu1_p(RGB_cash)
            u_p2 = self.unc_mu2_p(NI_cash)
            u_p3 = self.unc_mu3_p(TI_cash)

            
            s_p1 = self.unc_logvar1_p(RGB_cash)
            s_p2 = self.unc_logvar1_p(NI_cash)
            s_p3 = self.unc_logvar1_p(TI_cash)


            u1 = torch.cat([u_c1.unsqueeze(1), u_p1], dim=1)
            u2 = torch.cat([u_c2.unsqueeze(1), u_p2], dim=1)
            u3 = torch.cat([u_c3.unsqueeze(1), u_p3], dim=1)

            s1 = torch.cat([s_c1.unsqueeze(1), s_p1], dim=1)
            s2 = torch.cat([s_c2.unsqueeze(1), s_p2], dim=1)
            s3 = torch.cat([s_c3.unsqueeze(1), s_p3], dim=1)

            adjR = self.GraphConstructR(u1)
            adjN = self.GraphConstructN(u2)
            adjT = self.GraphConstructT(u3)

            eye = torch.eye(adjR.size(1), device=adjR.device).unsqueeze(0).expand_as(adjR)  
            
            normalized_adjR = normalize_adj(adjR + eye.clone().detach())
            normalized_adjN = normalize_adj(adjN + eye.clone().detach())
            normalized_adjT = normalize_adj(adjT + eye.clone().detach())
            u1 = self.MessagePassingR(u1, normalized_adjR)
            u2 = self.MessagePassingN(u2, normalized_adjN)
            u3 = self.MessagePassingT(u3, normalized_adjT)

            s1 = self.MessagePassingR1(s1, normalized_adjR)
            s2 = self.MessagePassingN1(s2, normalized_adjN)
            s3 = self.MessagePassingT1(s3, normalized_adjT)

            s1 = torch.sigmoid(s1)
            s2 = torch.sigmoid(s2)
            s3 = torch.sigmoid(s3)
            

            z1 = self.reparameterise(u1, self.w[0] * s1)    
            z2 = self.reparameterise(u2, self.w[1] * s2)   
            z3 = self.reparameterise(u3, self.w[2] * s3)    

            RGB_global = z1[:, 0, :]
            NI_global = z2[:, 0, :] 
            TI_global = z3[:, 0, :] 
            RGB_local = z1[:, 1:, :] 
            NI_local = z2[:, 1:, :]
            TI_local = z3[:, 1:, :]

            RGB_local = 0.2*self.pool(RGB_local.permute(0, 2, 1)).squeeze(-1)  + 0.8*self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
            NI_local =  0.2*self.pool(NI_local.permute(0, 2, 1)).squeeze(-1)   + 0.8*self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
            TI_local =  0.2*self.pool(TI_local.permute(0, 2, 1)).squeeze(-1)   + 0.8*self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
            RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
            NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
            TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))

        RGB_special = RGB_global 
        NI_special = NI_global
        TI_special = TI_global

        if self.training:
            if self.GPGR and not self.UGMoE:
                moe_feat = torch.cat(
                    [RGB_special, NI_special, TI_special], dim=-1)
                return moe_feat, 0
            if not self.GPGR and  self.UGMoE:
                moe_feat, loss_reg = self.forward_ATM(RGB_special, NI_special, TI_special)
                # return moe_feat, loss_reg
                # un_loss = 0
                return moe_feat, loss_reg
            elif not self.GPGR and not self.UGMoE:
                moe_feat = torch.cat(
                    [RGB_special, NI_special, TI_special], dim=-1)
                return moe_feat, 0
            elif self.GPGR and self.UGMoE:
                
                moe_feat, loss_reg = self.forward_UGMoE(RGB_special, NI_special, TI_special)
                return moe_feat, loss_reg + un_loss
        else:
            if self.GPGR and not self.UGMoE:
                moe_feat = torch.cat(
                    [RGB_special, NI_special, TI_special], dim=-1)
                if moe_feat.dim() == 1:
                    moe_feat = moe_feat.unsqueeze(0)
                return moe_feat
            elif not self.GPGR and  self.UGMoE:
                moe_feat = self.forward_ATM(RGB_special, NI_special, TI_special)
                return moe_feat
            elif not self.GPGR and not self.UGMoE:
                moe_feat = torch.cat(
                    [RGB_special, NI_special, TI_special], dim=-1)
                if moe_feat.dim() == 1:
                    moe_feat = moe_feat.unsqueeze(0)
                return moe_feat
            elif self.GPGR and self.UGMoE:
                moe_feat = self.forward_UGMoE(RGB_special, NI_special, TI_special)
                return moe_feat
