import numpy as np
from strlearn.metrics import balanced_accuracy_score
from mde import STML, STML_RGB2
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import strlearn as sl
from tqdm import tqdm
import matplotlib.pyplot as plt


random_state = 1410
n_chunks = 1000
chunk_size = 250

stream = sl.streams.StreamGenerator(n_chunks=n_chunks, chunk_size=chunk_size, y_flip=.05,
                                                            n_drifts=30, concept_sigmoid_spacing=None, incremental=False, recurring=False, 
                                                            n_features=8, n_informative=8, n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                                                            random_state=1410)

print("Start: %s" % (stream))
results = []

"""
Model
"""
num_classes = 2
batch_size = 8
num_epochs = 1
# weights = ResNet18_Weights.IMAGENET1K_V1
weights = None

model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("mps")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
criterion = nn.CrossEntropyLoss()
"""
"""

for c in tqdm(range(n_chunks)):
    X, y = stream.get_chunk()
    
    stml = STML(size=(50, 50))
    X_stml = stml.fit_transform(X)

    # stml = STML_RGB2(size=(50, 50))
    # X_stml = stml.fit_transform(X, y)

    X_stml = torch.from_numpy(np.moveaxis(X_stml, 3, 1)).float()
    y_stml = torch.from_numpy(y).long()
    
    stml_dataset = TensorDataset(X_stml, y_stml)
    data_loader = DataLoader(stml_dataset, batch_size=batch_size, shuffle=True)
    
    if c==0:
        model.train()
        for epoch in range(num_epochs):
            for i, batch in enumerate(data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                
    else:
        model.eval()
        logits = model(X_stml.to(device))
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        preds = np.argmax(probs, 1)
        score =  balanced_accuracy_score(y_stml.numpy(), preds)
        # scores = [metric(y_stml.numpy(), preds) for metric in metrics]
        results.append(score)
        
        model.train()
        for epoch in range(num_epochs):
            for i, batch in enumerate(data_loader, 0):
                inputs, labels = batch

                optimizer.zero_grad()

                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()

results = np.array(results)
np.save("results/stream/%s_binary" % stream, results)
print("End: %s" % (stream))