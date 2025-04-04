import numpy as np
from utils import Data
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score
from modAL.models import ActiveLearner
from mde import STML
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.decomposition import PCA


random = np.random.RandomState(1410)

# data = Data(selection=["haberman"], path="datasets/")
data = Data(selection=['australian', 'banknote', 'breastcancoimbra', 'cryotherapy', 'german', 'haberman', 'heart', 'ionosphere', 'liver', 'mammographic', 'monk-2', 'monkone', 'phoneme', 'pima', 'ring', 'sonar', 'spambase', 'titanic', 'twonorm', 'wisconsin'], path="datasets/")
datasets = data.load()
budget = .1
# DATASETS x FOLDS x METHOD x QUERIES
scores = np.zeros((len(datasets), 10, 3, 3000))

for data_id, dataset_name in enumerate(tqdm(datasets)):
    X, y = datasets[dataset_name][0], datasets[dataset_name][1]
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
    
    for fold_id, (train_index, test_index) in enumerate(tqdm(rskf.split(X, y), leave=False, desc=f"{data_id}", total=10, disable=False)):
        pool_index = random.choice(train_index, int((1-budget)*len(train_index)), replace=False)
        train_index = np.array([i for i in train_index if i not in pool_index])

        X_train, y_train = X[train_index], y[train_index]
        X_pool, y_pool = X[pool_index], y[pool_index]
        X_test, y_test = X[test_index], y[test_index]

        X_pool_cross = np.copy(X_pool)
        y_pool_cross = np.copy(y_pool)

        n_queries = int(X_pool.shape[0]/2)

        clf = XGBClassifier(n_estimators = 100)
        al_clf = ActiveLearner(estimator=clf, X_training=X_train, y_training=y_train)
                               
        # STML
        stml = STML()
        X_train_stml = torch.from_numpy(np.moveaxis(stml.fit_transform(X_train), 3, 1)).float()
        y_train_stml = torch.from_numpy(y_train).long()
        X_pool_stml = torch.from_numpy(np.moveaxis(stml.fit_transform(X_pool), 3, 1)).float()
        y_pool_stml = torch.from_numpy(y_pool).long()
        X_test_stml = torch.from_numpy(np.moveaxis(stml.fit_transform(X_test), 3, 1)).float()
        y_test_stml = torch.from_numpy(y_test).long()

        y_pool_stml_cross = torch.clone(y_pool_stml)

        num_classes = 2
        batch_size = 8
        n_epochs= 20

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device("mps")
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(X_train_stml, y_train_stml)
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

        model.eval()
        
        return_nodes = {
            'flatten': 'extracted_flatten',
        }
        extractor = create_feature_extractor(model, return_nodes=return_nodes)
        X_train_stml_extracted = extractor(X_train_stml.to(device))["extracted_flatten"].cpu().detach().numpy()
        X_pool_stml_extracted = extractor(X_pool_stml.to(device))["extracted_flatten"].cpu().detach().numpy()
        X_test_stml_extracted = extractor(X_test_stml.to(device))["extracted_flatten"].cpu().detach().numpy()

        X_pool_stml_extracted_cross = np.copy(X_pool_stml_extracted)

        # pca= PCA(n_components=.7, random_state=1410)
        # X_train_stml_extracted = pca.fit_transform(X_train_stml_extracted)
        # X_pool_stml_extracted = pca.transform(X_pool_stml_extracted)
        # X_test_stml_extracted = pca.transform(X_test_stml_extracted)

        # print(X_train_stml_extracted.shape)
        # print(X_pool_stml_extracted.shape)
        # print(X_test_stml_extracted.shape)
        # exit()

        clf_stml = XGBClassifier(n_estimators = 100)
        al_clf_stml = ActiveLearner(estimator=clf_stml, X_training=X_train_stml_extracted, y_training=y_train_stml)

        clf_cross_tab = XGBClassifier(n_estimators = 100)
        al_clf_cross_tab= ActiveLearner(estimator=clf_cross_tab, X_training=X_train, y_training=y_train)
        # clf_cross_stml = XGBClassifier(n_estimators = 100)
        # al_clf_cross_stml= ActiveLearner(estimator=clf_cross_stml, X_training=X_train_stml_extracted, y_training=y_train_stml)

        # Active learning
        for idx in range(n_queries):
            # TAB
            query_idx, query_instance = al_clf.query(X_pool)
            al_clf.teach(X_pool[query_idx], y_pool[query_idx])
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)
            preds = al_clf.predict(X_test)
            score = balanced_accuracy_score(y_test, preds)

            scores[data_id, fold_id, 0, idx] = score

            # STML
            query_idx, query_instance = al_clf_stml.query(X_pool_stml_extracted)
            al_clf_stml.teach(X_pool_stml_extracted[query_idx], y_pool_stml[query_idx])
            X_pool_stml_extracted = np.delete(X_pool_stml_extracted, query_idx, axis=0)
            y_pool_stml = np.delete(y_pool_stml, query_idx)
            preds = al_clf_stml.predict(X_test_stml_extracted)
            score = balanced_accuracy_score(y_test_stml, preds)
            scores[data_id, fold_id, 1, idx] = score

            # CROSS
            # query_idx, query_instance = al_clf_cross_stml.query(X_pool_stml_extracted_cross)
            al_clf_cross_tab.teach(X_pool_cross[query_idx], y_pool_cross[query_idx])
            # al_clf_cross_stml.teach(X_pool_stml_extracted_cross[query_idx], y_pool_stml_cross[query_idx])
            # X_pool_stml_extracted_cross = np.delete(X_pool_stml_extracted_cross, query_idx, axis=0)
            # y_pool_stml_cross = np.delete(y_pool_stml_cross, query_idx)
            X_pool_cross = np.delete(X_pool_cross, query_idx, axis=0)
            y_pool_cross = np.delete(y_pool_cross, query_idx)
            preds = al_clf_cross_tab.predict(X_test)
            score = balanced_accuracy_score(y_test, preds)
            scores[data_id, fold_id, 2, idx] = score

        np.save("results/active/preliminary_full", scores)
