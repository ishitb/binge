import utils
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np

def loss_fn(o1,o2,t1,t2):
    l = nn.CrossEntropyLoss()
    
    loss_s = l(o1,t1)
    loss_e = l(o2,t2)
    return loss_s+loss_e

    
def train_fn(data_loader,model,optimizer,device,scheduler):
    model.train()
    losses = utils.AverageMeter()
    jaccard = utils.AverageMeter()
    tk0 = tqdm(data_loader,total = len(data_loader))
    for bi,d in enumerate(tk0):
        ids = d['ids']
        offsets = d['offsets']
        orig_tweet = d['orig_tweet']
        orig_selected = d['orig_selected_text']
        token_type_ids = d['token_type_ids']
        sentiments = d['orig_sentiment']
        mask = d['mask']
        target_start = d['targets_start']
        target_end = d['targets_end']
       
                
        ids = ids.to(device,dtype = torch.long)
        token_type_ids = token_type_ids.to(device,dtype = torch.long)
        mask = mask.to(device,dtype = torch.long)
        target_start = target_start.to(device,dtype = torch.long)
        target_end = target_end.to(device,dtype = torch.long)
       
        
        model.zero_grad()    

        out_start,out_end = model(
                ids,
                mask,
                token_type_ids
                )
         
        loss = loss_fn(out_start,out_end,target_start,target_end)
        loss.backward()
        optimizer.step()
        scheduler.step()

        out_start = torch.softmax(out_start,dim = 1).cpu().detach().numpy()
        out_end = torch.softmax(out_end,dim = 1).cpu().detach().numpy()
        
            
        jac_scores = []
       # print(sentiment)
       # print(offsets,len(offsets),type(offsets))
        for j,tweet in enumerate(orig_tweet):
          #  print(j)
            offset = offsets[j]
            selected_text = orig_selected[j]
            sentiment = sentiments[j]
            idx_start = np.argmax(out_start[j,:])
            idx_end = np.argmax(out_end[j,:])
            _,jac = utils.calculate_jaccard(tweet,offset,selected_text,
                                            idx_start,idx_end,sentiment)
            jac_scores.append(jac)
            
       
            
        jaccard.update(np.mean(jac_scores),ids.size(0))
        losses.update(loss.item(),ids.size(0))
        tk0.set_postfix(loss = losses.avg,jaccard = jaccard.avg)
        
def eval_fn(data_loader,model,device):
    model.eval()
    losses = utils.AverageMeter()
    jaccard = utils.AverageMeter()
    with torch.no_grad():
        losses = utils.AverageMeter()
        jaccard = utils.AverageMeter()
        tk0 = tqdm(data_loader,total = len(data_loader))
        for bi,d in enumerate(tk0):
            ids = d['ids']
            offsets = d['offsets']         
            orig_selected = d['orig_selected_text']
            token_type_ids = d['token_type_ids']
            sentiments = d['orig_sentiment']
            mask = d['mask']
            target_start = d['targets_start']
            target_end = d['targets_end']
            orig_tweet = d['orig_tweet']
            
            
            ids = ids.to(device,dtype = torch.long)
            token_type_ids = token_type_ids.to(device,dtype = torch.long)
            mask = mask.to(device,dtype = torch.long)
            target_start = target_start.to(device,dtype = torch.long)
            target_end = target_end.to(device,dtype = torch.long)
           
            out_start,out_end = model(
                    ids,
                    mask,
                    token_type_ids
                    )
             
            loss = loss_fn(out_start,out_end,target_start,target_end)
    
            out_start = torch.softmax(out_start,dim = 1).cpu().detach().numpy()
            out_end = torch.softmax(out_end,dim = 1).cpu().detach().numpy()
           # print(out_start.shape,out_end.shape)
            jac_scores = []
            # print(offsets,len(offsets),type(offsets))
            for j,tweet in enumerate(orig_tweet):
                offset = offsets[j]
                selected_text = orig_selected[j]
                idx_start = np.argmax(out_start[j,:])
                sentiment = sentiments[j]
                idx_end = np.argmax(out_end[j,:])
                _,jac = utils.calculate_jaccard(tweet,offset,selected_text,
                                                idx_start,idx_end,sentiment)
                jac_scores.append(jac)
                
            jaccard.update(np.mean(jac_scores),ids.size(0))
            losses.update(loss.item(),ids.size(0))
            tk0.set_postfix(loss = losses.avg,jaccard = jaccard.avg)
                            
            return np.mean(jac_scores)            
        
