import torch
from torch import nn,optim

import torchvision
from torchvision import datasets, transforms, models

import numpy as np
from tqdm import tqdm
#################################################################################################
# Train and test functions

def train_fn(model,dataloader,loss_fn,optimizer,current_epo,epochs,device="cuda"):
    
    # Put model in train mode
    model.train()
        
    # Setup train loss and train accuracy values
    total_loss, total_acc = 0, 0
    tk=tqdm( dataloader,desc=("Ep"+"[Train]"+str(current_epo+1)+"/"+str(epochs)) )

    for t,data in enumerate(tk):
        tokens,txt,labels=data
        txts = {k: v.to(device) for k, v in tokens.items()}
        labels = labels.to(device)
        
        y_pred=model(txts)

        loss=loss_fn(y_pred,labels)
        total_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        total_acc += (y_pred_class == labels).sum().item()/len(y_pred)

        tk.set_postfix({"loss":"%.3f" %float(total_loss/(t+1)),"acc":"%.4f" %float(total_acc/(t+1))})

    return total_loss/len(dataloader), total_acc/len(dataloader)

    
def eval_fn(model,dataloader,loss_fn,current_epo,epochs,device='cuda',phase="eval"):

    model.eval()
    total_loss=0.0
    total_acc=0.0
    
    tk=tqdm(dataloader, desc=("Epoch"+"[Test]"+str(current_epo+1)+"/"+str(epochs)) )

    for t,data in enumerate(tk):
        tokens,txt,labels=data
        txts = {k: v.to(device) for k, v in tokens.items()}
        labels = labels.to(device)
        
        y_pred=model(txts)

        loss=loss_fn(y_pred,labels)
        total_loss+=loss.item()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        total_acc += (y_pred_class == labels).sum().item()/len(y_pred)
        tk.set_postfix({"loss":"%.3f" %float(total_loss/(t+1)),"acc":"%.4f" %float(total_acc/(t+1))})

    return (total_loss/len(dataloader)),(total_acc/len(dataloader))
    
def main(model,Trainloader,Devloader,Testloader,loss_fn,optimizer,lr,name=None,epochs=15,
         Test=False,scheduler_bol=False,save_weights=False,device='cuda'):

    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, cooldown=0, verbose=True)
    if not Test:
        best_dev_loss=np.Inf
        valid_losses=[]
                
        for i in range(epochs):
           
            train_loss , train_acc=train_fn(model, Trainloader, loss_fn, optimizer , i,epochs ,device)
            dev_loss   , dev_acc  =eval_fn(model,  Devloader,   loss_fn,              i,epochs,device,phase="eval")
            
            if scheduler_bol:
                scheduler.step(dev_loss)

            if dev_loss <best_dev_loss:
                if save_weights:
                    torch.save(model.state_dict(),f"./Saved_weights/{name}_best_weights.pt")
                #print('best weights saved')
                best_dev_loss=dev_loss
                
    Test_loss , Test_acc=eval_fn (model, Testloader,   loss_fn, 0,1,device,phase="test")
        
        