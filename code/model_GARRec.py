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
        
        # Save warm and cold item info (for later use if needed)
        self.warm_item = warm_item  
        self.cold_item = cold_item  

        # Common embedding table for users and items.
        self.id_embedding = nn.Parameter(
            nn.init.xavier_normal_(torch.rand(num_user + num_item, dim_E))
        )
        
        # Build content feature extractor for items.
        # If any of the features is provided, we concatenate them.
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
            # Assume features are aligned by item index (without user shift)
            self.content_feat = torch.cat(content_list, dim=1)
            content_dim = self.content_feat.size(1)
            # Generator: map content features into the same embedding space.
            self.generator = nn.Sequential(
                nn.Linear(content_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, dim_E)
            )
        else:
            self.content_feat = None
            self.generator = None

        # Discriminator: an MLP to judge user-item pair quality.
        # It takes concatenated user and item embeddings.
        self.discriminator = nn.Sequential(
            nn.Linear(dim_E * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def feature_extractor(self):
        # If a generator is defined, use it to generate embeddings from content.
        if self.generator is not None and self.content_feat is not None:
            # Note: self.content_feat should be indexed by item (without user shift)
            gen_emb = self.generator(self.content_feat)
            return gen_emb
        else:
            return None

    def forward(self, user_tensor, item_tensor):
        # Standard forward pass: compute recommendation scores using dot product.
        # item_tensor is assumed to contain positive and negative items.
        user_emb = self.id_embedding[user_tensor]  # shape: (batch, dim_E)
        item_emb = self.id_embedding[item_tensor]   # shape: (batch, dim_E)
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

    def loss(self, user_tensor, item_tensor):
        """
        This method returns a tuple (loss, reg_loss) similar to CLCRec.
        For adversarial loss, we use the discriminator to differentiate between
        real item embeddings (from id_embedding) and generated ones (from content features).
        
        Here, we assume that item_tensor's first column is the positive item.
        """
        # Get embeddings for users and the positive items.
        user_emb = self.id_embedding[user_tensor]  
        pos_item_tensor = item_tensor[:, 0]  # positive items
        pos_item_emb = self.id_embedding[pos_item_tensor]
        
        # Generate "fake" item embeddings from content for cold items.
        if self.generator is not None and self.content_feat is not None:
            # Here we assume that the content features are ordered by the original item id.
            # Since in id_embedding items are shifted by num_user, adjust indices accordingly.
            gen_emb_all = self.feature_extractor()  # shape: (num_item, dim_E)
            # Gather generated embeddings for the positive items.
            fake_item_emb = gen_emb_all[pos_item_tensor - self.num_user]
        else:
            fake_item_emb = pos_item_emb  # fallback if no generator
        
        # Build pairs for the discriminator.
        # Real pair: (user_emb, pos_item_emb)
        real_pair = torch.cat([user_emb, pos_item_emb], dim=1)
        # Fake pair: (user_emb, fake_item_emb)
        fake_pair = torch.cat([user_emb, fake_item_emb], dim=1)
        
        # Discriminator outputs
        d_real = self.discriminator(real_pair)
        d_fake = self.discriminator(fake_pair)
        
        # Adversarial losses:
        # For discriminator: want to output 1 for real pairs, 0 for fake pairs.
        loss_d_real = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
        loss_d_fake = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        d_loss = loss_d_real + loss_d_fake
        
        # For generator: want the discriminator to classify fake pairs as real.
        g_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
        
        # Combine losses. You can weight them using self.contrastive if desired.
        combined_loss = self.contrastive * g_loss + (1 - self.contrastive) * d_loss
        
        # Regularization on embeddings (similar to CLCRec)
        reg_loss = self.reg_weight * (torch.norm(user_emb, p=2) + torch.norm(pos_item_emb, p=2)) / 2
        
        return combined_loss, reg_loss
