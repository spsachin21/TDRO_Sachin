import torch
import torch.nn as nn
import math
import scipy.optimize as sopt
import torch.nn.functional as F
from torch.autograd import grad

def train_ERM(dataloader, model, optimizer):    
    model.train()
    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()
        loss = model.loss(user_tensor.cuda(), item_tensor.cuda())
        loss.backward()
        optimizer.step()
    return loss

def train_TDRO(dataloader, model, optimizer_d, optimizer_g, adj_matrix, 
               n_group, n_period, loss_list, w_list, mu, eta, lamda, beta_p):
    model.train()
    m = nn.Softmax(dim=1)
    beta_e = m(torch.tensor([math.exp(beta_p * e) for e in range(n_period)])
               .unsqueeze(0).unsqueeze(-1).cuda())
    
    adv_coef = 0.5  # From previous input, Amazon
    int_coef = 0.6  # From previous input, Amazon
    
    for user_tensor, item_tensor, group_tensor, period_tensor in dataloader:
        user_ids = user_tensor[:, 0]
        batch_size = user_ids.size(0)
        candidates = adj_matrix[user_ids].cuda()
        
        neg_item_ids_list = []
        log_prob_sum = 0
        num_negatives = 5
        for _ in range(num_negatives):
            neg_ids, log_prob = model.select_negatives(user_ids.cuda(), candidates)
            neg_item_ids_list.append(neg_ids)
            log_prob_sum += log_prob
        neg_item_ids = torch.stack(neg_item_ids_list, dim=1)  # [batch_size, num_negatives]
        
        pos_item_ids = item_tensor[:, 0]
        sample_loss, reg_loss = model.loss(user_ids.cuda(), pos_item_ids.cuda(), neg_item_ids[:, 0].cuda())
        
        loss_ge = torch.zeros((n_group, n_period)).cuda()
        grad_ge = torch.zeros((n_group, n_period, 
                              model.id_embedding.shape[0] * model.id_embedding.shape[1])).cuda()
        
        for g_idx in range(n_group):
            for e_idx in range(n_period):
                indices = ((group_tensor.squeeze(1) == g_idx) & (period_tensor.squeeze(1) == e_idx))
                de = torch.sum(indices)
                if de > 0:
                    loss_single = torch.sum(sample_loss[indices.cuda()])
                    grad_single = grad(loss_single, model.id_embedding, retain_graph=True)[0].reshape(-1)
                    grad_single = grad_single / (grad_single.norm() + 1e-16) * torch.pow(loss_single / (de + 1e-16), 1)
                    loss_ge[g_idx, e_idx] = loss_single
                    grad_ge[g_idx, e_idx] = grad_single
        
        de = torch.tensor([torch.sum(group_tensor == g_idx) for g_idx in range(n_group)]).cuda()
        loss_ = torch.sum(loss_ge, dim=1) / (de + 1e-16)
        
        trend_ = torch.zeros(n_group).cuda()
        for g_idx in range(n_group):
            g_j = torch.mean(grad_ge[g_idx], dim=0)
            sum_gie = torch.mean(grad_ge * beta_e, dim=[0, 1])
            trend_[g_idx] = g_j @ sum_gie
        
        loss_ = loss_ * (1 - lamda) + trend_ * lamda  # lamda=0.9
        
        loss_[loss_ == 0] = loss_list[loss_ == 0]
        loss_list = (1 - mu) * loss_list + mu * loss_
        
        update_factor = eta * loss_list * 2
        w_list = w_list * torch.exp(update_factor)
        w_list = w_list / torch.sum(w_list)
        loss_weightsum = int_coef * (torch.sum(w_list * loss_list) / batch_size) + reg_loss / batch_size
        
        optimizer_d.zero_grad()
        loss_weightsum.backward()
        optimizer_d.step()
        
        optimizer_g.zero_grad()
        user_emb = model.id_embedding[user_ids.cuda()]
        neg_emb_mean = torch.mean(model.id_embedding[neg_item_ids.cuda()], dim=1)
        reward = torch.sum(user_emb * neg_emb_mean, dim=1)
        reinforce_loss = adv_coef * (-torch.mean(reward * log_prob_sum) / num_negatives)
        reinforce_loss.backward()
        optimizer_g.step()
        
        loss_list.detach_()
        w_list.detach_()

    with torch.no_grad():
        model.result[model.emb_id] = model.id_embedding[model.emb_id].data
        model.result[model.feat_id + model.num_user] = model.feature_extractor()[model.feat_id].data if model.feature_extractor() is not None else model.id_embedding[model.feat_id + model.num_user].data

    return loss_weightsum
