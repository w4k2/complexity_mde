import numpy as np
from tqdm import tqdm
# PyTorch
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
# Sources
from mde import STML
from utils import Data
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification


n_epochs = 20

# data = Data(selection=("all", ["balanced", "binary"]), path="datasets/")
data = Data(selection=['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin']
, path="datasets/")
datasets = data.load()

for data_id, dataset_name in enumerate(tqdm(datasets)):
    X_, y_ = datasets[dataset_name][0], datasets[dataset_name][1]

    X, y = make_classification(n_samples=2000, n_features=X_.shape[1], n_informative=X_.shape[1], n_redundant=0, n_repeated=0, random_state=1410)

    # pca = PCA(n_components=4, random_state=1410)
    # X = pca.fit_transform(X)
        
    # Encoding        
    stml = STML()

    X_encoded = stml.fit_transform((X))
    X_encoded = torch.from_numpy(np.moveaxis(X_encoded, 3, 1)).float()
    y = torch.from_numpy(y).long()
    
     # Model
    num_classes = 2
    batch_size = 8
    # weights = None
    weights = ResNet18_Weights.IMAGENET1K_V1
    
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("mps")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(X_encoded, y)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in tqdm(range(n_epochs), leave=False):
        model.train()
        
        for i, batch in enumerate(train_data_loader, 0):
            inputs, labels = batch

            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            loss = criterion(outputs.to(device), labels.to(device))
            # loss_agg.append(loss.item())
            loss.backward()
            optimizer.step()

    torch.save(model, "models/model_%s_synth.pt" % dataset_name)
