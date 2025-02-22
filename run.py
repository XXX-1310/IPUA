import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from tensorflow import keras
from models import MFBasedModel
from itertools import cycle
from collections import defaultdict

class Run():
    def __init__(self,
                 config
                 ):
        self.use_cuda = config['use_cuda']
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_aug = self.batchsize_src
        self.c_num = 30
        self.topk = 12
        self.alph = 0.9

        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.num_fields = config['num_fields']
        self.lr = config['lr']
        self.wd = config['wd']

        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'

        self.results = {'ptupcdr_mae': 10, 'ptupcdr_rmse': 10}

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)#s u-i [1697533, 2] t u-i [1002084, 2]
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20, padding='post')
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter

    def read_map_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)#0到用户数量
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_aug_data(self):
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        # print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data()
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        # data_aug = self.read_aug_data()#将源域和目标域的训练数据拼接在一起
        # print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        return data_src, data_tgt, data_map,data_meta,data_test

    def get_model(self):
        if self.base_model == 'MF':
            model = MFBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim,self.c_num,self.topk)
        # elif self.base_model == 'DNN':
        #     model = DNNBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        # elif self.base_model == 'GMF':
        #     model = GMFBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_int = torch.optim.Adam(params=model.int_model.parameters(), lr=self.lr, weight_decay=self.wd)
        all_parameters = list(model.src_model.parameters()) + list(model.tgt_model.parameters()) + list(model.int_model.parameters())
        optimizer_mos = torch.optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_all = torch.optim.Adam(params=all_parameters, lr=self.lr, weight_decay=self.wd)
        optimizer_meta = torch.optim.Adam(params=model.meta_net.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer_src, optimizer_tgt, optimizer_int,optimizer_all,optimizer_mos

    def eval_mae(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def retain_topk(self, uim,k):
        topk_values, topk_indices = torch.topk(uim, k=k, dim=1)
        result = torch.zeros_like(uim)
        result.scatter_(1, topk_indices, topk_values)
        return result
    
    def creat_pseudo(self,sim,tim,pseudou_indices,pseudou_values,c_num,k):
        #一次
        common_groups = defaultdict(lambda: ([], []))
        user_interest_matrix_1 = self.retain_topk(sim,k)
        user_interest_matrix_2 = self.retain_topk(tim,k)
        for interest_id in range(c_num):
            domain1_users = torch.nonzero(user_interest_matrix_1[:, interest_id]).squeeze()
            domain2_users = torch.nonzero(user_interest_matrix_2[:, interest_id]).squeeze()   
            if domain1_users.numel() > 0 and domain2_users.numel() > 0:
                common_groups[interest_id] = (domain1_users, domain2_users)
        #两次
        #interest_group_representations = self.model.share_intent_embedding
        for interest_id, (domain1_users, domain2_users) in common_groups.items():
            if domain1_users.numel() > 0 and domain2_users.numel() > 0:
                #H_k = interest_group_representations.weight.data[interest_id]  # 该兴趣组的表征向量
                domain2_users = domain2_users[domain2_users != 0]#暴力方案
                # 获取域1和域2的用户特征矩阵
                e_it_matrix = user_interest_matrix_1[domain1_users,interest_id].cpu()  # 域1用户特征矩阵 (n1, d)
                e_il_matrix = user_interest_matrix_2[domain2_users,interest_id].cpu() # 域2用户特征矩阵 (n2, d)
                # 计算相似性矩阵 (n1, n2)并归一化
                similarity_matrix = torch.abs(e_it_matrix.unsqueeze(1) - e_il_matrix.unsqueeze(0))
                row_norms = similarity_matrix.norm(p=2, dim=1, keepdim=True)
                row_norms[row_norms == 0] = 1e-8
                similarity_matrix = similarity_matrix / row_norms
                # 找到最相关的索引及对应的值
                min_similarity_values, min_indices = similarity_matrix.min(dim=1)  # 最小值及其索引
                min_similarity_values, min_indices = min_similarity_values.cuda(), min_indices.cuda()
                pseudou_indices[domain1_users, interest_id] = domain2_users[min_indices]
                pseudou_values[domain1_users, interest_id] = min_similarity_values
        return pseudou_indices,pseudou_values
  

    def cotrain(self, data_loader1, data_loader2, data_loader3, model, criterion,kl_loss, optimizer, epoch, stage):
        print('Training Epoch {}:'.format(epoch + 1))
        w1,w2,w3 = self.datar_weights(data_loader1, data_loader2, data_loader3)
        model.train()
        for (X1, y1), (X2, y2),(X3, y3) in tqdm.tqdm(zip(data_loader1, cycle(data_loader2),cycle(data_loader3)), smoothing=0, mininterval=1.0):
            pred1 = model(X1, stage="src_interst")
            loss1 = criterion(pred1, y1.squeeze().float())
            pred2 = model(X2, stage="tgt_interst")
            loss2 = criterion(pred2, y2.squeeze().float())
            src_emb, tgt_emb = model(X3, stage= "overlap_training")
            loss3 = kl_loss(src_emb, tgt_emb)
            loss = w1*loss1 + w2*loss2 + w3*loss3
            #loss = loss1 + loss2 + loss3
            model.zero_grad()
            loss.backward()
            optimizer.step()

    def semitrain(self, data_loader1, data_loader2,model, criterion,criterion_none,optimizer, epoch, stage):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for (X1, y1), (X2, y2) in tqdm.tqdm(zip(data_loader1, cycle(data_loader2)), smoothing=0, mininterval=1.0):
            #伪标签训练
            src_emb, tgt_emb = model(X1, stage)
            loss1 = criterion(src_emb, tgt_emb)
            #真实重叠标签训练
            # src_emb, tgt_emb, weight = model(X2, stage)
            # lossa = criterion_none(src_emb, tgt_emb).mean(dim=-1)#128*8
            pred2 = model(X2, stage="test_meta")
            # lossb = criterion_none(pred2, y2.squeeze().float())
            # #动态平衡两个损失函数
            # adjusted_weight = torch.exp(-weight*1e6)  # 权重越低，贡献越高
            # #normalized_weight = adjusted_weight / adjusted_weight.sum()  # 归一化权重
            # lambda1 = adjusted_weight / (adjusted_weight+ 1)  # loss1 的动态权重128*8
            # lambda2 = 1 - lambda1  # loss2 的动态权重
            # loss2 = (lambda1 * lossa).mean(dim=-1) + lambda2.mean(dim=-1) * lossb
            loss2 = criterion(pred2, y2.squeeze().float())
            # loss = 0.1*loss1 + loss2.mean()
            loss = self.alph*loss1 + (1-self.alph)*loss2
            model.zero_grad()
            loss.backward()
            optimizer.step()
    
    def pseudou(self, data_loader1, data_loader2, model,stage):
        pseudou_index = torch.zeros(self.uid_all, self.c_num).long().cuda()
        pseudou_value = torch.zeros(self.uid_all, self.c_num).float().cuda()
        sim = torch.zeros(self.uid_all, self.c_num).cuda()
        tim = torch.zeros(self.uid_all, self.c_num).cuda()
        with torch.no_grad():
            for X1, _ in tqdm.tqdm(data_loader1, smoothing=0, mininterval=1.0):
                p1 = model(X1, stage="sim_learning")
                sim[X1[:,0]] = p1
            for X2, _ in tqdm.tqdm(data_loader2, smoothing=0, mininterval=1.0):
                p2 = model(X2, stage="tim_learning")
                tim[X2[:,0]] = p2
        model.pseudou_index, model.pseudou_value = self.creat_pseudo(sim,tim,pseudou_index,pseudou_value,self.c_num,self.topk)
    
    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse

    def datar_weights(self,data_loader1, data_loader2, data_loader3):
        # 假设 data_loader1, data_loader2, data_loader3 已经定义
        N1 = len(data_loader1.dataset)
        N2 = len(data_loader2.dataset)
        N3 = len(data_loader3.dataset)

        # 计算权重系数
        w1 = 1 / N1
        w2 = 1 / N2
        w3 = 1 / N3
        total_weight = w1 + w2 + w3
        w1 /= total_weight
        w2 /= total_weight
        w3 /= total_weight
        return w1,w2,w3


    def CDR(self, model, data_src, data_tgt, data_map,data_meta,data_test,
            criterion,criterion_none,kl_loss,optimizer_all, optimizer_meta):
        print('==========interst_pool==========')
        for i in range(self.epoch):
            self.cotrain(data_src, data_tgt,data_map,model, criterion,kl_loss,optimizer_all, i, stage='train_interst')          
            mae, rmse = self.eval_mae(model, data_test, stage='test_interst')
            self.update_results(mae, rmse, 'ptupcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))
        print('==========pseudou_users==========')
        self.pseudou(data_src,data_tgt,model,stage="")
        print('==========interst_bridge==========')
        for i in range(self.epoch):
            self.semitrain(data_src,data_meta,model, criterion,criterion_none,optimizer_meta, i, stage='train_meta')
            mae, rmse = self.eval_mae(model, data_test, stage='test_map')
            self.update_results(mae, rmse, 'ptupcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def main(self):
        model = self.get_model()
        #model = torch.load("task3_0.8.pth")
        data_src, data_tgt, data_map,data_meta,data_test = self.get_data()
        optimizer_src, optimizer_tgt, optimizer_int,optimizer_all,optimizer_meta = self.get_optimizer(model)
        criterion = torch.nn.MSELoss()#
        criterion_none = torch.nn.MSELoss(reduction='none')#
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        #self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)#只使用目标域数据训练的模型
        #self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)#增强模型
        self.CDR(model, data_src, data_tgt, data_map,data_meta,data_test,
                 criterion,criterion_none,kl_loss,optimizer_all, optimizer_meta)
        #torch.save(model, "task3_0.2.pth")
        # print("Complete model saved.")
        print(self.results)
