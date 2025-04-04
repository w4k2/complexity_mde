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
from mde import STML
from utils import Data


dataset_names = ['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']

# DATASET x FOLDS x TRANSFER
scores = np.load("results/transfer/prelim_bac_full.npy")
transrates = np.load("results/transfer/prelim_transrates_full.npy")

# Remove breastcan and bupa (2, 4)
del_datasets = list(np.delete(np.arange(scores.shape[0]), [2, 4]))
del_transfer = list(np.delete(np.arange(scores.shape[2]), [3, 5]))
scores = scores[del_datasets]
scores = scores[:, :, del_transfer]
transrates = transrates[del_datasets]
transrates = transrates[:, :, del_transfer]
# DATASET x TRANSFER
mean_scores = np.mean(scores, axis=1).squeeze()
mean_transrates = np.mean(transrates, axis=1).squeeze()

"""
Choose the best models for each dataset (not imagenet)
"""
best_model_bac = []
best_model_transrate = []

for data_id, data_name in enumerate(dataset_names):
    tmp_names = np.delete(np.array(dataset_names), [data_id])

    data_bac = mean_scores[data_id][1:]
    data_bac_best = np.argsort(-np.delete(data_bac, [data_id]))
    best_model_bac.append([tmp_names[data_bac_best][0]])

    data_transrate = mean_transrates[data_id][1:]
    data_transrate_best = np.argsort(-np.delete(data_transrate, [data_id]))
    best_model_transrate.append([tmp_names[data_transrate_best][0]])

best_model_bac = np.array(best_model_bac)
best_model_transrate = np.array(best_model_transrate)

"""
Experiment
"""


data = Data(selection=dataset_names, path="datasets/")
datasets = data.load()

transfer_names = np.concatenate((np.array(["imagenet" for i in range(best_model_bac.shape[0])]).reshape(-1, 1), best_model_bac, best_model_transrate), axis=1)

# Results
# DATASETS x FOLDS x TRANSFER (imagenet | best BAC | best transrate) x EPOCH
n_epochs = 50
results = np.zeros((len(datasets), 10, 3, n_epochs))

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
        
        for transfer_id, transfer in enumerate(transfer_names[data_id]):
            if transfer_id == 2 and transfer_names[data_id][2] == transfer_names[data_id][1]:
                pass
            else:
                # Model
                num_classes = 2
                batch_size = 8
                

                if transfer == "imagenet":
                    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                else:
                    model = torch.load("models/model_%s_imgn.pt" % transfer, weights_only=False)

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
                    
                    results[data_id, fold_id, transfer_id, epoch] = balanced_accuracy_score(y_test, preds)
                        
                np.save("results/transfer/comparison", results)