import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_cifar import resnet18_cifar
from utils import identify_importance
import numpy as np
import copy
# from .helper import *
import timm
import torch.nn as nn
# from timm.models import vit_base_patch16_224_in21k
from models.vision_transformer import VisionTransformer
# 1. 从 timm 导入原始模块
from timm.models.vision_transformer import Attention
from timm.models.layers import trunc_normal_
#todo PKT for domain specific knowledge learning..
#todo PKT with B-Prompt ==> Prefix Tuning 
#todo Need Something to focus on domain specific knowledge learning 
#todo finc inciteness from the Novel Category Discovery 
# import open_clip as clip

class CustomAttention(Attention):
    def forward(self, x, prompt=None):
        # import pdb 
        # pdb.set_trace()
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            if type(prompt) is list and len(prompt) == 3:
                pq, pk, pv = prompt
                
                q_new = q.clone()
                k_new = k.clone()
                v_new = v.clone()
                
                
                
                # 在副本上进行修改
                q_new[:, :, 0:1] = q_new[:, :, 0:1] + pq[:, :, 0:1]
                k_new[:, :, 0:1] = k_new[:, :, 0:1] + pk[:, :, 0:1]
                v_new[:, :, 0:1] = v_new[:, :, 0:1] + pv[:, :, 0:1]
                
                # 使用修改后的副本
                q = q_new
                k = k_new
                v = v_new
                
                # k[:,:,0:1] = k[:,:,0:1] + pk[:,:,0:1]   #加到 cls token的 key上  每个头都加
                # v[:,:,0:1] = v[:,:,0:1] + pv[:,:,0:1]   #加到 cls token的value上 每个头都加
            else:
                raise ValueError("prompt type not supported!")
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CustomViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args 
        #encoder
        self.encoder = timm.create_model("vit_base_patch16_224",pretrained=True,num_classes=args.num_classes,
                                drop_rate=0.,drop_path_rate=0.,drop_block_rate=None)
        self.encoder.head = nn.Identity()
        dim = 768
        
        self.changeBlocks()
            
        #prompt
        self.prompt = APT(768,12)
        #cls
        self.cls_prompt = nn.Parameter(torch.zeros(768))
        #fc
        self.num_tokens = 197
        self.num_heads = self.encoder.blocks[0].attn.num_heads
        self.fc = nn.Linear(dim, self.args.num_classes, bias=False)
        self.seen_classes = args.base_class
    
    def changeBlocks(self):
        for i, block in enumerate(self.encoder.blocks):
            original_attn = block.attn
            #现有attn替换代码
            # 安全获取维度属性
            if hasattr(original_attn, 'embed_dim'):
                dim = original_attn.embed_dim
            else:
                dim = original_attn.qkv.in_features  # 或 original_attn.qkv.weight.shape[1]
            
            custom_attn = CustomAttention(
                dim = dim,
                num_heads = original_attn.num_heads,
                qkv_bias=original_attn.qkv.bias is not None,
                attn_drop=original_attn.attn_drop.p,
                proj_drop=original_attn.proj_drop.p
            )
            # 复制权重
            custom_attn.load_state_dict(original_attn.state_dict())
            block.attn = custom_attn
            
            #新增：修改block的forwaed方法
            original_block_forward = block.forward
            
            def custom_block_forward(_self, x, prompt=None):
                # 处理第一个子层（Attention）
                if prompt is not None:
                    attn_result = _self.attn(_self.norm1(x), prompt)
                else:
                    attn_result = _self.attn(_self.norm1(x))
                
                x = x + _self.drop_path1(attn_result)
                
                # 处理第二个子层（MLP）
                x = x + _self.drop_path2(_self.mlp(_self.norm2(x)))
                return x
            
            # 绑定新方法到当前block
            block.forward = custom_block_forward.__get__(block)
        
    def forward_prompt_embed_pre(self,x,layer_stop = -1): #-1
        x = self.encoder.patch_embed(x)
        cls_up = self.encoder.cls_token + self.cls_prompt
        ex_cls = cls_up.expand(x.shape[0], -1, -1) 
        #ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)  #位置编码
        add_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
        
        if layer_stop == -1:
            return x
        
        for i, blk in enumerate(self.encoder.blocks):
            if i in add_layers:
                prompt_list = self.prompt.forward(i,x,train=True)
                x = blk(x, prompt_list)
                if i == layer_stop:
                    break
            else:
                x = blk(x)  # 不传递prompt
        return x
    
    def forward_prompt_embed_post(self, x, layer_start = 0): #0
        add_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
        
        for i, blk in enumerate(self.encoder.blocks):
            if i < layer_start:
                continue
            if i in add_layers:
                prompt_list = self.prompt.forward(i,x,train=True)
                x = blk(x, prompt_list)
            else:
                x = blk(x)  # 不传递prompt
        x = self.encoder.norm(x)
        
        cls_embed = x[:,0]
        
        logit = F.linear(F.normalize(cls_embed, p=2, dim=-1), F.normalize(self.fc.weight, p=2,dim=-1) )
        logit = self.args.temperature*logit
        return logit
    
    def forward_prompt_embed(self,x):
        
        x = self.encoder.patch_embed(x)
        cls_up = self.encoder.cls_token + self.cls_prompt
        ex_cls = cls_up.expand(x.shape[0], -1, -1) 
        #ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)  #位置编码
        
        add_layers = [0,1,2,3,4,5,6,7,8,9,10,11]
        #x = self.encoder.blocks(x)
        for i, blk in enumerate(self.encoder.blocks):
            if i in add_layers:
                prompt_list = self.prompt.forward(i,x,train=True)
                x = blk(x, prompt_list)
            else:
                x = blk(x)  # 不传递prompt
        x = self.encoder.norm(x)
        return x
    
    def forward_train_prompt(self, x):
        x = self.forward_prompt_embed(x)
        cls_embed = x[:,0]
        
        logit = F.linear(F.normalize(cls_embed, p=2, dim=-1), F.normalize(self.fc.weight, p=2,dim=-1) )
        logit = self.args.temperature*logit
        return logit,cls_embed
    
    def forward_val_prompt(self, x):
        x = self.forward_prompt_embed(x)
        cls_embed = x[:,0]
        
        logit = F.linear(F.normalize(cls_embed, p=2, dim=-1), F.normalize(self.fc.weight, p=2,dim=-1) )
        logit = self.args.temperature*logit
        return logit
    
    def forward_embed_prompt(self, x):
        x = self.forward_prompt_embed(x)
        cls_embed = x[:,0]
        
        # logit = F.linear(F.normalize(cls_embed, p=2, dim=-1), F.normalize(self.fc.weight, p=2,dim=-1) )
        # logit = self.args.temperature*logit
        return cls_embed

    
    #all forward
    def forward_train(self, x):
        x = self.encoder.patch_embed(x)
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)  #位置编码
        
        x = self.encoder.blocks(x)
        cls_embed = x[:,0]
        
        logit = F.linear(F.normalize(cls_embed, p=2, dim=-1), F.normalize(self.fc.weight, p=2,dim=-1) )
        logit = self.args.temperature*logit
        return logit,cls_embed
    
    def forward_val(self, x):
        x = self.encoder.patch_embed(x)
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)  #位置编码
        
        x = self.encoder.blocks(x)
        cls_embed = x[:,0]
        
        logit = F.linear(F.normalize(cls_embed, p=2, dim=-1), F.normalize(self.fc.weight, p=2,dim=-1) )
        logit = self.args.temperature*logit
        return logit
    
    def forward_embed(self, x):
        x = self.encoder.patch_embed(x)
        ex_cls = self.encoder.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat([ex_cls,x],dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)  #位置编码
        
        x = self.encoder.blocks(x)
        cls_embed = x[:,0]
        
        return cls_embed
    
    #replace fc
    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.mask = torch.zeros(self.args.num_classes,device='cuda')
        self.mask[:self.seen_classes]=-torch.inf
        self.seen_classes += len(new_classes)
    
    def update_fc(self, dataloader,class_list,session):
        feats = []
        labels = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                data,label = [_.cuda() for _ in batch]
                #self.mode = 'encoder'
                embedding = self.forward_embed_prompt(data)
                feats.append(embedding.cpu())
                labels.append(label.cpu())
        feats = torch.cat(feats,dim=0)
        labels = torch.cat(labels,dim=0)
        
        for ii in range(labels.unique().shape[0]):
            self.fc.weight.data[labels.min()+ii] = feats[labels==labels.min()+ii].mean(dim=0)
    
    def train_inc(self, dataloader, epochs, session, class_list):
        if epochs == 0:
            return 
        print("[Session: {}]".format(session))
        #self.update_fc_avg(dataloader, class_list, query_info)      #把新类原型加入到之前类的原型中，更新fc
        self.train()
        
        if epochs > 0: 
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr_new)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
            
        for idx,batch in enumerate(dataloader):

            data_imgs, data_label = [_.cuda() for _ in batch]
            #optim = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), momentum=0.9, lr=0.01, weight_decay = 0.005   )
            
            for epoch in range(epochs):
                logits,cls_embed = self.forward_train_prompt(data_imgs)
                
                loss_ce = F.cross_entropy(logits, data_label)
                
                loss = loss_ce
                optim.zero_grad()
                loss.backward()
                
                optim.step()
                scheduler.step()
                pred = torch.argmax(logits, dim=1)
                acc = (pred == data_label).sum().item()/data_label.shape[0]*100.
                print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} ACC: {acc}")
                # if self.args.SKD:
                #     print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} loss_kb:{loss_kb.item():.4f} ACC: {acc}")
                # else:
                #     print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} ACC: {acc}")

class APT(nn.Module):
    def __init__(self, emb_d,  layers, dropout_rate=0.2):
        super().__init__()
        self.emb_d = emb_d

        #self.prompt_tokens = self.create_prompt_with_init(layers*2, emb_d) 
        
        self.prompt_tokens = self.create_prompt_with_init(layers*3, emb_d)
        
        #global_merged_prompt = torch.zeros(12*2, emb_d).cuda()
        #self.register_buffer('global_merged_prompt', global_merged_prompt.clone().detach()) 

        trunc_normal_(self.prompt_tokens, std=0.02)
        self.dropouts = nn.Dropout(p=dropout_rate)

        # for i in range(12):
        #     setattr(self, f'k_layer_proj{i}', nn.Linear(2, 2))
        #     setattr(self, f'v_layer_proj{i}', nn.Linear(2, 2))
        
    def create_prompt_with_init(self, a, b, c=None, ortho=False, mean=None, std=None, init_ref=None):
        if c is None:
            p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
        
        if ortho:
            nn.init.orthogonal_(p)
        elif init_ref is not None:
            p = torch.nn.Parameter(init_ref.squeeze(dim=0).expand(a, b),  requires_grad=True)
        elif mean and std:
            nn.init.normal_(p, mean=mean, std=std)
        else:
            nn.init.uniform_(p)
        return p
        
   
    def merge_prompt(self, prompt1, prompt2):
        print("Merging prompt ... ")
        return prompt1*self.ema_coeff + prompt2*(1-self.ema_coeff)

    def process_task_count(self):
        self.task_count += 1

    def forward(self, l, x_block, train=False):
        B, _, _ = x_block.shape

        prompt_groups = self.prompt_tokens
        
        if train or not self.merge_flag:  #训练模式
            P_root_q = self.dropouts(prompt_groups[l*3:l*3+1]).reshape(12,1,64).expand(B,12,1,64)    #该层的cls_k, 768->[12,1,64]->[B,12,1,64]
            P_root_k = self.dropouts(prompt_groups[l*3+1:l*3+2]).reshape(12,1,64).expand(B,12,1,64)  #该层的cls_v, 768->[12,1,64]->[B,12,1,64]
            P_root_v = self.dropouts(prompt_groups[l*3+2:l*3+3]).reshape(12,1,64).expand(B,12,1,64)  #该层的cls_v, 768->[12,1,64]->[B,12,1,64]
        elif not train and self.merge_flag: #推理模式
            pass
            # P_root_q = self.global_merged_prompt[l*2:l*2+1].reshape(12,1,64).expand(B,12,1,64)  #
            # P_root_k = self.global_merged_prompt[l*2+1:l*2+2].reshape(12,1,64).expand(B,12,1,64)
            # P_root_v = self.global_merged_prompt[l*2+2:l*2+3].reshape(12,1,64).expand(B,12,1,64)
        else:
            raise ValueError("merge flag and mode err")

        P_q = torch.cat((P_root_q, torch.zeros((B,12,196,64),device =x_block.device)),dim=-2)
        P_k = torch.cat((P_root_k, torch.zeros((B,12,196,64),device =x_block.device)),dim=-2)
        P_v = torch.cat((P_root_v, torch.zeros((B,12,196,64),device =x_block.device)),dim=-2)
        
        P = [P_q,P_k, P_v]    

        return P #, rpt_index
    
    def freeze(self):
        """确保所有提示参数被冻结"""
        for param in self.parameters():
            param.requires_grad = False

