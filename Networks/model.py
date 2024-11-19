from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.UNetSemanticSegmenter import *
from Networks.Architectures.basicNetwork import *

import numpy as np
np.random.seed(2885)
import os
import copy

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim

from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
import copy


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = UNetSmnticSgmntr(in_ch=3, inside_ch=param["MODEL"]["NB_CHANNEL"],depth=1,threshold=0.5).to(self.device)
        #self.model = Net(param).to(self.device)

        # -------------------
        # TRAINING PARAMETERS
        # -------------------
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) 

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = OxfordPetDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal      = OxfordPetDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = OxfordPetDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)


    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl'))

    # -----------------------------------
    # TRAINING LOOP (fool implementation)
    # -----------------------------------
    def train(self): 
        
        tresh = 0.5
        train_losses, val_losses = [], []
        t_accuracies, t_recalls, t_precisions = [], [], []
        v_accuracies, v_recalls, v_precisions = [], [], []

        for i in range(self.epoch):

            tmp_accuracies, tmp_recalls, tmp_precisions = [], [], []

            #Train mode
            self.model.train()
            train_loss = 0.0

            #Training loop
            for images, masks, _ in self.trainDataLoader:
                images, masks = images.to(self.device), masks.to(self.device)
                masks = masks.unsqueeze(1).float()
                #Rétropropagation
                self.optimizer.zero_grad()
                #Update weights
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                #Compute metrics
                outputs = outputs.detach().cpu().numpy()
                masks = masks.cpu().numpy()
                out = (outputs > tresh).astype(np.int32)

                acc = accuracy_score(masks.flatten(), out.flatten())
                tmp_accuracies.append(acc)

                rec = recall_score(masks.flatten(), out.flatten())
                tmp_recalls.append(rec)

                prec = precision_score(masks.flatten(), out.flatten())
                tmp_precisions.append(prec)
                


            avg_train_loss = train_loss / len(self.trainDataLoader)
            train_losses.append(avg_train_loss)

            t_accuracies.append(np.mean(tmp_accuracies))
            t_recalls.append(np.mean(tmp_recalls))
            t_precisions.append(np.mean(tmp_precisions))

            tmp_accuracies, tmp_recalls, tmp_precisions = [], [], []

            #Evaluation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks, _ in self.valDataLoader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    masks = masks.unsqueeze(1).float()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    val_loss += loss.item()

                    outputs = outputs.cpu().numpy()
                    masks = masks.cpu().numpy()

                    #Compute metrics
                    out = (outputs > tresh).astype(np.int32)

                    acc = accuracy_score(masks.flatten(), out.flatten())
                    tmp_accuracies.append(acc)

                    rec = recall_score(masks.flatten(), out.flatten())
                    tmp_recalls.append(rec)

                    prec = precision_score(masks.flatten(), out.flatten())
                    tmp_precisions.append(prec)

            #Average loss
            avg_val_loss = val_loss / len(self.valDataLoader)
            val_losses.append(avg_val_loss)

            v_accuracies.append(np.mean(tmp_accuracies))
            v_recalls.append(np.mean(tmp_recalls))
            v_precisions.append(np.mean(tmp_precisions))

            print(f"Loss at {i}-th epoch: ")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Train - Accuracy: {t_accuracies[-1]:.4f}, "
                f"Recall: {t_recalls[-1]:.4f}, "
                f"Precision: {t_precisions[-1]:.4f}")
            print(f"  Val   - Accuracy: {v_accuracies[-1]:.4f}, "
                f"Recall: {v_recalls[-1]:.4f}, "
                f"Precision: {v_precisions[-1]:.4f}")

            modelWts = copy.deepcopy(self.model.state_dict())

        # Save the model weights
        wghtsPath  = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(modelWts, wghtsPath + '/wghts.pkl')

        self.plot(train_losses, val_losses)
        self.plot(t_accuracies, v_accuracies)



    def plot(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    # -------------------------------------------------
    # EVALUATION PROCEDURE (ultra basic implementation)
    # -------------------------------------------------
    def evaluate(self):
        self.model.train(False)
        self.model.eval()
        
        # Qualitative Evaluation 
        allInputs, allPreds, allGT = [], [], []
        for (images, GT, resizedImg) in self.testDataLoader:
            images      = images.to(self.device)
            predictions = self.model.pred(images, 0.5)

            images, predictions = images.to('cpu'), predictions.to('cpu')

            allInputs.extend(resizedImg.data.numpy())
            allPreds.extend(predictions.data.numpy())
            allGT.extend(GT.data.numpy())

        allInputs = np.array(allInputs)
        allPreds  = np.array(allPreds)
        allGT     = np.array(allGT)

        showPredictions(allInputs, allPreds, allGT, self.resultsPath)

        # Quantitative Evaluation
        # Implement this ! 

        IoU_scores = []
        for pred, GT in zip(allPreds, allGT):
            
            pred = pred.flatten()
            GT = GT.flatten()

            IoU = jaccard_score(GT, pred)
            IoU_scores.append(IoU)

        mean_iou = np.mean(IoU_scores)

        print(f"Quantitative Evaluation: {mean_iou:.4f}")
        