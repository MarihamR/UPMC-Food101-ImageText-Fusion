import random 
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def plot_samples(split_set,preds=None,name=None):

    idx_to_class = {split_set.class_to_idx[k]: k for k in split_set.class_to_idx}
    randomlist = random.sample(range(0, len(split_set)), 9)
    #print(randomlist)
    k=0
    figure, ax = plt.subplots(3, 3,constrained_layout = True)

    for i in range(3):
        for j in range(3):
            idx=randomlist[k]
            k+=1
            ax[i, j].imshow(split_set[idx][0].permute(1,2,0))
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)

            if  (  (preds is None) ):
                ax[i, j].set_title(f"{split_set[idx][1]}-{idx_to_class[split_set[idx][1]]}", fontsize=12)

            else:
                if split_set[idx][1]==preds[idx]:
                    check="Correctly_Classified"
                else:
                    check="Wrongly_Classified"

                ax[i, j].set_title(f"{check} \n Actual:{idx_to_class[split_set[idx][1]]} \n predicted:{idx_to_class[preds[idx]]}"
                               , fontsize=10)
            if name is not None:
                figure.savefig(f"{name}.png", dpi=300)
          
        

	    
def plot_confusion(labels, preds,name,num_classes=2,Normalize=False):
	if Normalize:
		conf=confusion_matrix(labels, preds,normalize='true')
	else:
		conf=confusion_matrix(labels, preds)
	
	if num_classes==2:
		labels_name=['Normal', 'Pneumonia']
	elif num_classes==3:
		labels_name=['Covid','Normal', 'Pneumonia']


	fig, ax = plt.subplots(figsize=(10,6))
	ax = sns.heatmap(conf, annot=True,xticklabels=labels_name,yticklabels=labels_name,fmt='.3f', 
		         cmap=sns.cubehelix_palette(as_cmap=True))

	font1 = {'family': 'sans-serif','weight': 'bold','color':"sienna",'size': 16}
	ax.set_xlabel('Predicted Label',fontdict=font1)
	ax.set_ylabel('Actual Label',fontdict=font1)
	plt.show()
	fig.savefig(f"./Visualizations/{name}.png", dpi=300)
    
def plot_samples2(split_set,preds=None,name=None):

    Classes=sorted(os.listdir('./datasets/Food101/images/test'))
    classes_idx=[i for i in range(101)]
    idx_to_class =dict (zip (classes_idx, Classes))
    randomlist = random.sample(range(0, len(split_set)), 9)
    #print(randomlist)
    k=0
    figure, ax = plt.subplots(3, 3,constrained_layout = True)

    for i in range(3):
        for j in range(3):
            idx=randomlist[k]
            k+=1
            ax[i, j].imshow(split_set[idx][0].permute(1,2,0))
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)

            if  (  (preds is None) ):
                ax[i, j].set_title(f"{split_set[idx][1]}-{idx_to_class[split_set[idx][1]]}", fontsize=12)

            else:
                if split_set[idx][1]==preds[idx]:
                    check="Correctly_Classified"
                else:
                    check="Wrongly_Classified"

                ax[i, j].set_title(f"{check} \n Actual:{idx_to_class[split_set[idx][1]]} \n predicted:{idx_to_class[preds[idx]]}"
                               , fontsize=10)
            if name is not None:
                figure.savefig(f"{name}.png", dpi=300)
                
                
                
def plot_Fusion_samples(split_set,preds=None,name=None):
    
    randomlist = random.sample(range(0, len(split_set)), 6)
    #print(randomlist)
    k=0
    figure, ax = plt.subplots(3, 2,constrained_layout = True)

    for i in range(3):
        for j in range(2):
            idx=randomlist[k]
            k+=1
            ax[i, j].imshow(split_set[idx][0].permute(1,2,0))
            ax[i,j].set_xlabel(f"Text: {split_set[idx][2]}",fontsize=8)
            ax[i,j].get_yaxis().set_visible(False)

            if  (  (preds is None) ):
                ax[i, j].set_title(f"Class:{split_set[idx][3]}-{split_set.idx_to_class[split_set[idx][3]]}", fontsize=12)

            else:
                if split_set[idx][3]==preds[idx]:
                    check="Correctly_Classified"
                else:
                    check="Wrongly_Classified"

                ax[i, j].set_title(f"{check} \n Actual:{split_set.idx_to_class[split_set[idx][3]]} \n predicted:{split_set.idx_to_class[preds[idx]]}"
                               , fontsize=10)
            if name is not None:
                figure.savefig(f"{name}.png", dpi=300)     
                           
