# Utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep
# scikit-learn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
# PyTorch
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
# Sources
from mde import STML, STML_RGB, STML_RGB2
from utils import Data
import matplotlib.pyplot as plt


num_epochs = 20


data = Data(selection=("all", ["balanced", "binary"]), path="datasets/")
datasets = data.load()

# Scores
# DATASETS x FOLDS x EPOCHS
scores = np.zeros((len(datasets), 10, num_epochs))
# DATASETS x TRAIN/TEST x FOLDS x EPOCHS
# losses = np.zeros((len(datasets), 2, 10, num_epochs))

for data_id, dataset_name in enumerate(tqdm(datasets)):
    X, y = datasets[dataset_name][0], datasets[dataset_name][1]
    # X = X
    # y = y
    
    # X_encoded = STML(X, size=(224, 224))
    
    # plt.imshow(X_stml[0])
    # plt.savefig("bar.png")
    # print(X.shape)
    # sleep(2)
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
    
    for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
        
        # Encoding
        # Train
        stml_rgb = STML_RGB2()
        X_encoded_train = stml_rgb.fit_transform(X[train_index], y[train_index])
        X_encoded_train = torch.from_numpy(np.moveaxis(X_encoded_train, 3, 1)).float()
        y_train = torch.from_numpy(y[train_index]).long()
        
        # Test
        X_encoded_test = stml_rgb.transform(X[test_index])
        X_encoded_test = torch.from_numpy(np.moveaxis(X_encoded_test, 3, 1)).float()
        y_test = torch.from_numpy(y[test_index]).long()
        
        # Model
        num_classes = 2
        batch_size = 8
        weights = ResNet18_Weights.IMAGENET1K_V1
        # weights = None
        
        model = resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        device = torch.device("mps")
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        train_dataset = TensorDataset(X_encoded_train, y_train)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in tqdm(range(num_epochs), leave=False, desc=f"{fold_id}"):
            model.train()
            # loss_agg = []
            
            for i, batch in enumerate(train_data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                # loss_agg.append(loss.item())
                loss.backward()
                optimizer.step()
                
            # losses[data_id, 0, fold_id, epoch] = np.mean(loss_agg)
        
            model.eval()
            logits = model(X_encoded_test.to(device))
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
            preds = np.argmax(probs, 1)
            
            # loss = criterion(logits.to(device), y_test.to(device))
            # losses[data_id, 1, fold_id, epoch] = loss.item()
            
            scores[data_id, fold_id, epoch] = balanced_accuracy_score(y_test, preds)
            
        np.save("results/ex00_preliminary_stml_rgb_transfer_224_scores", scores)