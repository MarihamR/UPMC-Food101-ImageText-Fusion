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

    #iterations in 1 epoch # iterations=len(trainloader)=901 batches each batch consist of32 images
    for t,data in enumerate(tk):
        images,labels=data
        images,labels=images.to(device) ,labels.to(device)


        # 1. Forward pass
        y_pred=model(images)


        # 2. Calculate  and accumulate loss
        loss=loss_fn(y_pred,labels)
        total_loss+=loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        total_acc += (y_pred_class == labels).sum().item()/len(y_pred)

     
        tk.set_postfix({"loss":"%.3f" %float(total_loss/(t+1)),"acc":"%.4f"%float(total_acc/(t+1))})

    return total_loss/len(dataloader), total_acc/len(dataloader)


def train_fn_without_displaying_epochs(model,dataloader,loss_fn,optimizer,current_epo,epochs,device='cuda'):
    
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    total_loss, total_acc = 0, 0
   
    for t,data in enumerate(dataloader):
        images,labels=data
        images,labels=images.to(device) ,labels.to(device)
        y_pred=model(images)
        loss=loss_fn(y_pred,labels)
        total_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        total_acc += (y_pred_class == labels).sum().item()/len(y_pred)

    return total_loss/len(dataloader), total_acc/len(dataloader)
    
def eval_fn(model,dataloader,loss_fn,current_epo,epochs,device='cuda',phase="eval"):

    model.eval()
    total_loss=0.0
    total_acc=0.0
    
    tk=tqdm(dataloader, desc=("Epoch"+"[Test]"+str(current_epo+1)+"/"+str(epochs)) )

    for t,data in enumerate(tk):
        images,labels=data
        images,labels=images.to(device) ,labels.to(device)

        y_pred=model(images)

        loss=loss_fn(y_pred,labels)
        total_loss+=loss.item()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        total_acc += (y_pred_class == labels).sum().item()/len(y_pred)
        tk.set_postfix({"loss":"%.3f" %float(total_loss/(t+1)),"acc":"%.4f"%float(total_acc/(t+1))})

    return (total_loss/len(dataloader)),(total_acc/len(dataloader))
    
def main(model,Trainloader,Devloader,Testloader,loss_fn,optimizer,lr,name=None,epochs=10,
         Test=False,scheduler_bol=False,nodisplay=False,sch_type="step",sch_gamma=0.1,
         sch_step=3,save_weights=False,device='cuda'):
    
    
    if sch_type=="step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=sch_step,gamma=sch_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=sch_gamma, patience=sch_step, cooldown=0, verbose=True)
    
    if not Test:
        best_valid_loss=np.Inf
        best_valid_acc=0
        valid_losses=[]
        for i in range(epochs):
            if not nodisplay:
                train_loss , train_acc=train_fn(model, Trainloader, loss_fn, optimizer , i,epochs,device)
                valid_loss , valid_acc=eval_fn (model, Devloader,   loss_fn,  i, epochs,device)
            else:
                train_loss , train_acc=train_fn_without_displaying_epochs(model, Trainloader, 
                                                                          loss_fn, optimizer , i,epochs ,device)
            if scheduler_bol==True :
                if sch_type=="step":
                    scheduler.step()
                else:
                    scheduler.step(valid_loss)

            if valid_loss <best_valid_loss:
                if save_weights:
                    torch.save(model.state_dict(),f"./Saved_weights/best_weights_{name}.pt")
                #print('best weights saved')
                best_valid_loss=valid_loss
                best_valid_acc=valid_acc
                

        #valid_losses.append(valid_loss)
        #print (f"best_valid_acc== {best_valid_acc}")
        if  nodisplay:
                print (f"Ep [Train] {str(epochs)}/{str(epochs)} :loss={best_train_loss}, acc={best_train_acc}")
            
        Test_loss , Test_acc=eval_fn (model, Testloader,   loss_fn, 0,1,device,phase="test")
    
    else:
        Test_loss , Test_acc=eval_fn (model, Testloader,   loss_fn, 0,1,device,phase="test")
        
        
def Test_ensemble(model1,model2,dataloader,loss_fn,device='cuda'):

    model1.eval()
    model2.eval()
    total_loss=0.0
    total_acc=0.0
    label=[]
    preds=[]
    

    tk=tk=tqdm(dataloader, desc=("Epoch"+"[Test]"+str(1)+"/"+str(1)) )


    for t,data in enumerate(tk):
        images,labels=data
        images,labels=images.to(device) ,labels.to(device)
        label.append(labels.detach().cpu().numpy())

        y_pred1=model1(images)
        y_pred2=model2(images)
        y_pred=y_pred1+y_pred2
        
        
        loss=loss_fn(y_pred,labels)
        total_loss+=loss.item()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        preds.append(y_pred_class.detach().cpu().numpy())
        
        total_acc += (y_pred_class == labels).sum().item()/len(y_pred)
        tk.set_postfix({"loss":"%3f" %float(total_loss/(t+1)),"acc":"%3f" %float(total_acc/(t+1))})
        
    label = [item for sublist in label for item in sublist]
    preds = [item for sublist in preds for item in sublist]
    
    return (total_loss/len(dataloader)),(total_acc/len(dataloader)),label ,preds
    
def Get_predictions(model,dataloader,device="cuda"):

    model.eval()
    labels=[]
    preds=[]
    
    for data in (dataloader):
        image,label=data
        image,label=image.to(device) ,label.to(device)
        labels.append(label.detach().cpu().numpy())

        y_pred=model(image)
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        preds.append(y_pred_class.detach().cpu().numpy())

    labels = [item for sublist in labels for item in sublist]
    preds = [item for sublist in preds for item in sublist]
    return labels,preds
