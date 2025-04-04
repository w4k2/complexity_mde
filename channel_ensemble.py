# Utils
import numpy as np
from tqdm import tqdm
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
from mde import STML, IGTD_npy, DeepInsight, Norm2Scaler
from utils import Data
import matplotlib.pyplot as plt


dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

data = Data(selection=dataset_names, path="datasets/")
datasets = data.load()

# Results
# DATASETS x FOLDS x EPOCH
n_epochs = 20
results = np.zeros((len(datasets), 10, n_epochs))

for data_id, dataset_name in enumerate(tqdm(datasets)):
    X, y = datasets[dataset_name][0], datasets[dataset_name][1]
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
    
    for fold_id, (train_index, test_index) in enumerate(tqdm(rskf.split(X, y), leave=False, desc=f"{data_id}", total=10)):
        # Encoding        
        # Train

        X_train = X[train_index]
        X_test = X[test_index]

        # STML
        stml = STML()
        X_encoded_train = stml.fit_transform((X_train))
        X_encoded_train_1 = torch.from_numpy(np.moveaxis(X_encoded_train, 3, 1)).float()
        X_encoded_test = stml.transform((X_test))
        X_encoded_test_1 = torch.from_numpy(np.moveaxis(X_encoded_test, 3, 1)).float()

        # IGTD
        X_encoded_train = IGTD_npy(X_train)
        X_encoded_train_2 = torch.from_numpy(np.moveaxis(X_encoded_train, 3, 1)).float()
        X_encoded_test = IGTD_npy(X_test)
        X_encoded_test_2 = torch.from_numpy(np.moveaxis(X_encoded_test, 3, 1)).float()

        # DI
        ln = Norm2Scaler()
        di = DeepInsight(feature_extractor='pca', 
        discretization='bin', pixels=(224, 224))
        X_encoded_train = ln.fit_transform(X[train_index])
        X_encoded_train = di.fit_transform(X_encoded_train)
        X_encoded_train_3 = torch.from_numpy(np.moveaxis(X_encoded_train, 3, 1)).float()

        X_encoded_test = ln.transform(X[test_index])
        X_encoded_test = di.transform(X_encoded_test)
        X_encoded_test_3 = torch.from_numpy(np.moveaxis(X_encoded_test, 3, 1)).float()

        X_encoded_train =torch.from_numpy(np.concatenate((X_encoded_train_1[:, :1], X_encoded_train_2[:, :1], X_encoded_train_3[:, :1]), axis=1))
        X_encoded_test = torch.from_numpy(np.concatenate((X_encoded_test_1[:, :1], X_encoded_test_2[:, :1], X_encoded_test_3[:, :1]), axis=1))

        # plt.imshow(np.moveaxis(X_encoded_train[0], 0, 2))
        # plt.savefig("foo.png")

        # y
        y_train = torch.from_numpy(y[train_index]).long()
        y_test = torch.from_numpy(y[test_index]).long()
        
        # Model
        num_classes = 2
        batch_size = 8
        
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device("mps")
        model = model.to(device)

        """
        Train
        """

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        train_dataset = TensorDataset(X_encoded_train, y_train)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in tqdm(range(n_epochs), leave=False, desc=f"{fold_id}"):
            model.train()
            
            for i, batch in enumerate(train_data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()

            """
            Test
            """

            model.eval()

            # Results
            logits = model(X_encoded_test.to(device))
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
            preds = np.argmax(probs, 1)
            
            results[data_id, fold_id, epoch] = balanced_accuracy_score(y_test, preds)
                
        np.save("results/channel_ensemble", results)