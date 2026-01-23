
# import new Network name here and add in model_class args
import time

# from .Network import MYNET

from utils import *
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertModel
import numpy as np
import torch


def replace_base_fc(trainset, transform, model, args):      #计算原型的时候，不加VLprompt和Base prompt
    print("[Replace Base FC - Original]")
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=4, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            embedding = model.module.forward_embed_prompt(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
        
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list      #把类别原型放入到fc中

    return model

def cross_entropy(preds, targets, reduction='none'):
    labels = torch.arange(targets.shape[0]).cuda()
    loss = F.cross_entropy(preds,labels, reduction='none')
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def base_train_my(model, trainloader, optimizer, scheduler, epoch, class_list, args):
    
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader, mininterval=1.0)
    
    
    # 定义软标签交叉熵损失函数
    def soft_cross_entropy(logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1).mean()
        return loss
    
    
    args.alpha = 2.0
    for i, batch in enumerate(tqdm_gen, 1):
        beta = torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        
        data, train_label = [_.cuda() for _ in batch]   
        logits, cls_embed = model.module.forward_train_prompt(data)
        logits = logits[:, :args.base_class]
        loss_ce = F.cross_entropy(logits, train_label)
        
        acc = count_acc(logits, train_label)
        
        # Manifold Mixup 部分
        index = torch.randperm(data.size(0)).cuda()
        pre_emb1 = model.module.forward_prompt_embed_pre(data, args.mix_layer)
        mixed_data = beta * pre_emb1 + (1 - beta) * pre_emb1[index]
        mixed_logits = model.module.forward_prompt_embed_post(mixed_data, args.mix_layer+1)
        
        # 获取混合样本的软标签（基于真实标签）
        with torch.no_grad():
            # 创建原始标签的one-hot编码
            train_label_onehot = F.one_hot(train_label, num_classes=args.base_class).float()
            # 创建混合样本的标签：混合两个样本的真实标签
            soft_targets = beta * train_label_onehot + (1 - beta) * train_label_onehot[index]
        
        # 使用软标签计算交叉熵损失
        loss_mix = soft_cross_entropy(mixed_logits[:,:args.base_class], soft_targets)
        
        # 总损失只包含原始分类损失和混合损失
        total_loss = loss_ce + args.mix*loss_mix
        
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        ta.add(acc, len(train_label))
        
        # tqdm_gen.set_description(
        #     'Session 0, epo {}, lrc={:.4f}, total loss={:.4f}, loss_CE={:.4f}, acc={:.4f}'.\
        #         format(epoch, lrc, total_loss.item(), loss_ce.item(), ta.item()))
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f}, total loss={:.4f}, loss_CE={:.4f}, loss_mix={:.4f},acc={:.4f}'.\
                format(epoch, lrc, total_loss.item(), loss_ce.item(), args.mix*loss_mix.item(), ta.item()))
        
        optimizer.zero_grad()   #梯度清零
        total_loss.backward()   #反向传播
        
        optimizer.step()        #随机梯度下降
    tl = tl.item()
    ta = ta.item()
    
    return tl, ta



def test(model, testloader, epoch, args, session):
    #todo Test시 Prompt Selection is needed..
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager_Loss()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
    print("\t\t\t[Test Phase] Session: {}".format(session))
    
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            logits = model.module.forward_val_prompt(data)     #加VL_prompt和Base_prompt
            logits = logits[:, :test_class]
            
            loss = F.cross_entropy(logits, test_label)
            
            acc = count_acc(logits, test_label)

            base_idxs = test_label < args.base_class   #筛选出test_label原本标签是旧类的
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs])   #
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs])
                va_base.add(acc_base, len(test_label[base_idxs]))
                va_base_given_new.add(acc_base_given_new, len(test_label[base_idxs]))


            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class)
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs])
                va_new.add(acc_new, len(test_label[new_idxs]))
                va_new_given_base.add(acc_new_given_base, len(test_label[new_idxs]))

            vl.add(loss.item(), len(test_label))
            va.add(acc, len(test_label))

        vl = vl.item()
        va = va.item()

        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
    print('base only accuracy: {:.4f}, new only accuracy: {:.4f}'.format(va_base, va_new))
    print('base acc given new : {:.4f}'.format(va_base_given_new))
    print('new acc given base : {:.4f}'.format(va_new_given_base))

    logs = dict(num_session=session + 1, acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new,
                new_acc_given_base=va_new_given_base)

    return vl, va, logs




            