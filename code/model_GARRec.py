import torch
import torch.nn as nn
import torch.nn.functional as F

class GARRec(nn.Module):
    def __init__(self, warm_item, cold_item, num_user, num_item, 
                 reg_weight, dim_E, v_feat, a_feat, t_feat, 
                 temp_value, num_neg, contrastive, num_sample):
        super(GARRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.reg_weight = reg_weight
        self.dim_E = dim_E
        self.temp_value = temp_value
        self.num_neg = num_neg
        self.contrastive = contrastive
        self.num_sample = num_sample

        # Save warm and cold item info
        self.warm_item = warm_item  
        self.cold_item = cold_item  

        # Attributes for TDRO
        self.emb_id = list(range(num_user)) + list(warm_item)
        self.feat_id = torch.tensor([i - num_user for i in cold_item])
        
        # Single learnable embedding table for users and items
        self.id_embedding = nn.Parameter(
            nn.init.xavier_normal_(torch.rand(num_user + num_item, dim_E))
        )
        
        # Content feature extractor for items (kept for cold-start functionality)
        self.v_feat = F.normalize(v_feat, dim=1) if v_feat is not None else None
        self.a_feat = F.normalize(a_feat, dim=1) if a_feat is not None else None
        self.t_feat = F.normalize(t_feat, dim=1) if t_feat is not None else None
        
        content_list = []
        if self.v_feat is not None:
            content_list.append(self.v_feat)
        if self.a_feat is not None:
            content_list.append(self.a_feat)
        if self.t_feat is not None:
            content_list.append(self.t_feat)
        if len(content_list) > 0:
            self.content_feat = torch.cat(content_list, dim=1)
            content_dim = self.content_feat.size(1)
            self.generator = nn.Sequential(
                nn.Linear(content_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, dim_E)
            )
        else:
            self.content_feat = None
            self.generator = None

        # Policy network for negative sampling
        self.policy_net = nn.Sequential(
            nn.Linear(dim_E, 256),
            nn.ReLU(),
            nn.Linear(256, dim_E)
        )
        
        # Result tensor for TDRO
        self.result = torch.zeros((num_user + num_item, dim_E)).cuda()

    def feature_extractor(self):
        if self.generator is not None and self.content_feat is not None:
            gen_emb = self.generator(self.content_feat)
            return gen_emb
        return None

    def forward(self, user_tensor, item_tensor):
        user_emb = self.id_embedding[user_tensor]
        item_emb = self.id_embedding[item_tensor]
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

    def loss(self, user_tensor, pos_item_tensor, neg_item_tensor):
        """
        Compute BPR loss using user, positive item, and negative item embeddings.
        
        Args:
            user_tensor: Tensor of user IDs [batch_size]
            pos_item_tensor: Tensor of positive item IDs [batch_size]
            neg_item_tensor: Tensor of negative item IDs [batch_size]
        
        Returns:
            sample_loss: Per-sample BPR loss [batch_size]
            reg_loss: Regularization loss
        """
        user_emb = self.id_embedding[user_tensor]        # [batch_size, dim_E]
        pos_item_emb = self.id_embedding[pos_item_tensor]  # [batch_size, dim_E]
        neg_item_emb = self.id_embedding[neg_item_tensor]  # [batch_size, dim_E]
        
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)  # [batch_size]
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)  # [batch_size]
        sample_loss = -F.logsigmoid(pos_scores - neg_scores)    # [batch_size]
        
        reg_loss = self.reg_weight * (
            torch.norm(user_emb, p=2) + 
            torch.norm(pos_item_emb, p=2) + 
            torch.norm(neg_item_emb, p=2)
        ) / 3
        
        return sample_loss, reg_loss

    def select_negatives(self, user_tensor, adj_matrix):
        user_emb = self.id_embedding[user_tensor]  # [batch_size, dim_E]
        policy_emb = self.policy_net(user_emb)     # [batch_size, dim_E]
        batch_size = user_tensor.size(0)
        num_candidates = adj_matrix.size(1)
        chunk_size = 32
        neg_item_ids = torch.zeros(batch_size, dtype=torch.long, device=user_tensor.device)
        log_prob = torch.zeros(batch_size, device=user_tensor.device)
        
        for i in range(0, num_candidates, chunk_size):
            end = min(i + chunk_size, num_candidates)
            chunk_candidates = adj_matrix[:, i:end]
            chunk_embs = self.id_embedding[chunk_candidates]
            chunk_logits = torch.sum(policy_emb.unsqueeze(1) * chunk_embs, dim=2)
            chunk_probs = F.softmax(chunk_logits, dim=1)
            dist = torch.distributions.Categorical(chunk_probs)
            chunk_idx = dist.sample()
            if i == 0:
                neg_item_ids = chunk_candidates[torch.arange(batch_size), chunk_idx]
                log_prob = dist.log_prob(chunk_idx)
            else:
                chunk_log_prob = dist.log_prob(chunk_idx)
                mask = chunk_log_prob > log_prob
                neg_item_ids = torch.where(mask, chunk_candidates[torch.arange(batch_size), chunk_idx], neg_item_ids)
                log_prob = torch.where(mask, chunk_log_prob, log_prob)
        
        return neg_item_ids, log_prob
