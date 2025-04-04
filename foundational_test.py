# Utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import sleep
from utils import transrate
# scikit-learn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
# PyTorch
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
# Sources
from mde import STML
from utils import Data



# data = Data(selection=['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin'], path="datasets/")
data = Data(selection=['australian', "banknote"], path="datasets/")
datasets = data.load()

# Scores
# DATASETS x FOLDS x TRANSFER
scores = np.zeros((len(datasets), 10, 1))
# DATASETS x FOLDS x TRANSFER
transrates = np.zeros((len(datasets), 10, 1))

for data_id, dataset_name in enumerate(tqdm(datasets)):
    X, y = datasets[dataset_name][0], datasets[dataset_name][1]
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
    
    for fold_id, (train_index, test_index) in enumerate(tqdm(rskf.split(X, y), leave=False, desc=f"{data_id}", total=10)):
        # Encoding        
        # Train
        stml = STML()

        X_train = X[train_index]
        X_test = X[test_index]

        X_encoded_train = stml.fit_transform((X_train))
        X_encoded_train = torch.from_numpy(np.moveaxis(X_encoded_train, 3, 1)).float()
        y_train = torch.from_numpy(y[train_index]).long()
        
        X_encoded_test = stml.transform((X_test))
        X_encoded_test = torch.from_numpy(np.moveaxis(X_encoded_test, 3, 1)).float()
        y_test = torch.from_numpy(y[test_index]).long()
        
        # Model
        num_classes = 2
        batch_size = 8

        
        model = torch.load("models/model_%s_foundational.pt" % dataset_name, weights_only=False)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device("mps")
        model = model.to(device)

        """
        """

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        train_dataset = TensorDataset(X_encoded_train, y_train)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in tqdm(range(20), leave=False, desc=f"{fold_id}"):
            model.train()
            # loss_agg = []
            
            for i, batch in enumerate(train_data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()

        """
        """


        model.eval()
        
        # SCORES
        logits = model(X_encoded_test.to(device))
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        preds = np.argmax(probs, 1)
        
        scores[data_id, fold_id, 0] = balanced_accuracy_score(y_test, preds)
        
        # TransRate
        return_nodes = {
            'flatten': 'extracted_flatten',
        }
        extractor = create_feature_extractor(model, return_nodes=return_nodes)
        X_extracted = extractor(X_encoded_test.to(device))["extracted_flatten"].cpu().detach().numpy()
        
        transrates[data_id, fold_id, 0] = transrate(X_extracted, y_test)
        
            
        np.save("results/transfer/foundational_test_scores", scores)
        np.save("results/transfer/foundational_test_transrates", transrates)