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
from mde import STML, DeepInsight, Norm2Scaler
from utils import Data


dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']
# dataset_names = ['australian', 'banknote']

"""
Experiment
"""
data = Data(selection=dataset_names, path="datasets/")
datasets = data.load()

# Results
# DATASETS x FOLDS x TRANSFER (imagenet | best BAC | best transrate) x EPOCH
n_epochs = 50
results = np.zeros((len(datasets), 10, 1, n_epochs))

for data_id, dataset_name in enumerate(tqdm(datasets)):
    # print(dataset_name)
    X, y = datasets[dataset_name][0], datasets[dataset_name][1]
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
    
    for fold_id, (train_index, test_index) in enumerate(tqdm(rskf.split(X, y), leave=False, desc=f"{data_id}", total=10)):
        # Encoding        
        # Train
        # stml = STML()
        ln = Norm2Scaler()
        di = DeepInsight(feature_extractor='pca', 
        discretization='bin', pixels=(224, 224))

        X_train = X[train_index]
        X_test = X[test_index]

        X_train = ln.fit_transform(X_train)
        X_encoded_train = di.fit_transform((X_train))
        X_encoded_train = torch.from_numpy(np.moveaxis(X_encoded_train, 3, 1)).float()
        y_train = torch.from_numpy(y[train_index]).long()
        
        X_test = ln.transform(X_test)
        X_encoded_test = di.transform((X_test))
        X_encoded_test = torch.from_numpy(np.moveaxis(X_encoded_test, 3, 1)).float()
        y_test = torch.from_numpy(y[test_index]).long()
        
        # Model
        num_classes = 2
        batch_size = 8


        model = torch.load("models/model_monkone_di.pt", weights_only=False)
        # Haberman i Monokone
        # model = torch.load("models/model_haberman_di_wo_imgnet.pt", weights_only=False)


        # Extraction or Fine-tuning
        # for param in model.parameters():
        #     param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device("cuda")
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
            
            results[data_id, fold_id, 0, epoch] = balanced_accuracy_score(y_test, preds)

        np.save("results2/transfer2/comparison_model_monkone_di_complexity", results)