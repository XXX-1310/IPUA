import torch
import torch.nn.functional as F
import math
import numpy as np


class LookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))#256，1，10
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)#256，2，10
        return emb
    
class InterstEmbedding(torch.nn.Module):

    def __init__(self,emb_dim,meta_dim,c_num,topk):
        super().__init__()
        self.embedding_size = meta_dim
        self.C = torch.nn.Embedding(c_num, meta_dim)
        self.initializer_range = 0.01
        self.w1 = self._init_weight((emb_dim, meta_dim))
        self.w2 = self._init_weight((meta_dim, emb_dim))
        self.k = topk
    
    def _init_weight(self, shape):
        mat = np.random.normal(0, self.initializer_range, shape)
        return torch.tensor(mat, dtype=torch.float32, requires_grad=True).cuda()
    
    def forward(self, x_u,stage):
        x = torch.matmul(x_u, self.w1)
        s_u = torch.matmul(self.C.weight, x.transpose(1, 2)).squeeze(2)
        # idx = s_u.argsort(1)[:, -self.k :]#256,3
        # s_u_idx = s_u.sort(1)[0][:, -self.k :]#256,3
        # c_u = self.C(idx)#256,3,10
        # sigs = torch.sigmoid(s_u_idx.unsqueeze(2).repeat(1, 1, self.embedding_size))
        # C_u = c_u.mul(sigs).sum(dim=1)
        # u_ins = torch.matmul(C_u, self.w2)
        gumbel_weights = F.gumbel_softmax(s_u, tau=10, hard=False, dim=-1)  #(batch_size, 100)
        topk_weights, topk_indices = torch.topk(gumbel_weights, self.k, dim=1)  # (batch_size, k)
        c_u = self.C(topk_indices)  # (batch_size, k, meta_dim)
        C_weighted = (c_u * topk_weights.unsqueeze(2)).sum(dim=1)  # (batch_size, meta_dim)
        u_ins = torch.matmul(C_weighted, self.w2)  # (batch_size, emb_dim)
        if stage in ['src_interst','tgt_interst','test_interst']:
            return u_ins
        elif stage in ['overlap_training','sim_learning','tim_learning']:
            return gumbel_weights
        elif stage in ['test_map','train_meta','test_meta']:
            return c_u,topk_weights


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Linear(60, 50), torch.nn.ReLU(),
                                           torch.nn.Linear(50, emb_dim * emb_dim))

    def forward(self, emb_fea):
        output = self.decoder(emb_fea)#torch.Size([256, 3, 1, 100])
        return output


class MFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0,c_num,topk):
        super().__init__()
        self.uid_all = uid_all
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.int_model = InterstEmbedding(emb_dim, meta_dim_0,c_num,topk)
        #self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.pseudou_index = None
        self.pseudou_value = None
        self.c_num = c_num
        self.topk = topk
    
    def forward(self, x, stage):
        if stage == 'train_src':
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage == 'src_interst':
            uid_emb = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))#256，1，10
            uid_ins = self.int_model.forward(uid_emb,stage).unsqueeze(1)#256,1,10
            iid_emb = self.src_model.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_ins, iid_emb], dim=1)#256，2，10
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage == 'tgt_interst':
            uid_emb = self.tgt_model.uid_embedding(x[:, 0].unsqueeze(1))#256，1，10
            uid_ins = self.int_model.forward(uid_emb,stage).unsqueeze(1)#256,1,10
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_ins, iid_emb], dim=1)#256，2，10
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage =='test_interst':
            uid_emb = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))#256，1，10
            uid_ins = self.int_model.forward(uid_emb,stage).unsqueeze(1)
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_ins, iid_emb], dim=1)#256，2，10
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage == 'overlap_training':
            # uid_src = self.src_model.uid_embedding(x.unsqueeze(1))#256，1，10
            # uid_tgt = self.tgt_model.uid_embedding(x.unsqueeze(1))#256，1，10
            # uid_sins = self.int_model.forward(uid_src)#256,10
            # uid_tins = self.int_model.forward(uid_tgt)#256,10
            # return uid_sins,uid_tins
            uid_src = self.src_model.uid_embedding(x.unsqueeze(1))#256，1，10 
            uid_tgt = self.tgt_model.uid_embedding(x.unsqueeze(1))#256，1，10
            p1 = self.int_model.forward(uid_src,stage)#256,10
            p2 = self.int_model.forward(uid_tgt,stage)#256,10
            return p1,p2
        elif stage == 'sim_learning':
            uid_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))#256，1，10
            p = self.int_model.forward(uid_src,stage)#256,10
            return p
        elif stage == 'tim_learning':
            uid_src = self.tgt_model.uid_embedding(x[:, 0].unsqueeze(1))#256，1，10
            p = self.int_model.forward(uid_src,stage)#256,10
            return p
        elif stage =='train_meta':           
            uid_emb = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))#128,1,10
            uis_matrix = self.pseudou_index[x[:, 0]]#128*50
            uis_values = self.pseudou_value[x[:, 0]]
            #
            iindict = torch.nonzero(uis_matrix)
            #weight = uis_values[iindict[:, 0], iindict[:, 1]].reshape(-1, self.topk)
            ins_fea,score_weight = self.int_model.forward(uid_emb,stage)
            src_emb = uid_emb.expand(-1, self.topk, -1)
            input_fea = torch.cat((ins_fea,src_emb),dim=-1)
            #
            mapping = self.meta_net.forward(input_fea.unsqueeze(-2)).view(-1, self.topk, self.emb_dim, self.emb_dim)#128，10，10，这里应该叫做mapping function Wu
            pre_emb = torch.matmul(src_emb.unsqueeze(-2), mapping).squeeze(-2)#128，k，10 
            uid_lab = uis_matrix[iindict[:, 0],iindict[:, 1]].view(-1, self.topk)#128*3
            tgt_emb = self.tgt_model.uid_embedding(uid_lab)#128*3*10
            return pre_emb,tgt_emb
        elif stage == 'test_meta':#test_map
            tuid_emb = self.tgt_model.uid_embedding(x[:, 0].unsqueeze(1))
            tuid_ins = self.int_model.forward(tuid_emb,stage='test_interst').unsqueeze(1)

            uid_emb = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))#128,1,10
            uid_ins, src_score = self.int_model.forward(uid_emb,stage)#256*3*50 256*3
            src_fea = uid_emb.expand(-1, self.topk, -1)
            input_fea = torch.cat((uid_ins,src_fea),dim=-1)
            mapping = self.meta_net.forward(input_fea.unsqueeze(-2)).view(-1, self.topk, self.emb_dim, self.emb_dim)#128，10，10，这里应该叫做mapping function Wu
            src_emb = torch.matmul(src_fea.unsqueeze(-2), mapping).squeeze(-2)#128，3，10
            pre_emb = (src_emb*(src_score.unsqueeze(-1))).sum(dim=1)
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))#128,1,10
            fin_emb = tuid_ins + pre_emb.unsqueeze(1)
            emb = torch.cat([fin_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage == 'test_map':#test_map
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))#128,1,10
            tuid_emb = self.tgt_model.uid_embedding(x[:, 0].unsqueeze(1))
            tuid_ins = self.int_model.forward(tuid_emb,stage='test_interst').unsqueeze(1)

            uid_emb = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))#128,1,10
            uid_ins, src_score = self.int_model.forward(uid_emb,stage)#256*3*50 256*3
            src_fea = uid_emb.expand(-1, self.topk, -1)
            #cross
            input_fea = torch.cat((uid_ins,src_fea),dim=-1)
            mapping = self.meta_net.forward(input_fea.unsqueeze(-2)).view(-1, self.topk, self.emb_dim, self.emb_dim)#128，10，10，这里应该叫做mapping function Wu
            src_emb = torch.matmul(src_fea.unsqueeze(-2), mapping).squeeze(-2)#128，3，10
            #
            pre_emb = (src_emb*(src_score.unsqueeze(-1))).sum(dim=1)
            fin_emb = tuid_ins + pre_emb.unsqueeze(1)
            emb = torch.cat([fin_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output

