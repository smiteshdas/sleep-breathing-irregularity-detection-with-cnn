

import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight




with open("Dataset/breathing_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)




unique_labels = sorted(list(set(d["label"] for d in dataset)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
num_classes = len(unique_labels)

print("Label Mapping:", label_to_int)




class CNN1D(nn.Module):
    def __init__(self, num_classes, input_channels, input_length):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            x = self.pool(torch.relu(self.conv1(dummy)))
            x = self.pool(torch.relu(self.conv2(x)))
            self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




participants = sorted(list(set(d["participant"] for d in dataset)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_accuracies = []
all_conf_matrices = []

for test_participant in participants:

    print(f"\nTesting on {test_participant}")

    train_data = [d for d in dataset if d["participant"] != test_participant]
    test_data  = [d for d in dataset if d["participant"] == test_participant]

    X_train = np.stack([d["signal"] for d in train_data])
    y_train = np.array([label_to_int[d["label"]] for d in train_data])

    X_test = np.stack([d["signal"] for d in test_data])
    y_test = np.array([label_to_int[d["label"]] for d in test_data])



    mean = X_train.mean(axis=(0,2), keepdims=True)
    std  = X_train.std(axis=(0,2), keepdims=True)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_test  = (X_test - mean) / std



    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )


    full_weights = np.ones(num_classes)
    present_classes = np.unique(y_train)

    for i, cls in enumerate(present_classes):
        full_weights[cls] = class_weights[i]

    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32).to(device)

    print("Class Weights:", full_weights)


    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_channels = X_train.shape[1]
    input_length   = X_train.shape[2]

    model = CNN1D(num_classes, input_channels, input_length).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)



    epochs = 15

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()



    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)

    y_pred = predictions.cpu().numpy()
    y_true = y_test.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    print("Accuracy:", acc)
    print("Macro Precision:", prec)
    print("Macro Recall:", rec)
    print("Confusion Matrix:\n", cm)

    all_accuracies.append(acc)
    all_conf_matrices.append(cm)




num_folds = len(all_conf_matrices)

fig, axes = plt.subplots(
    1,
    num_folds,
    figsize=(8 * num_folds, 4),
    constrained_layout=True
)
# 4, 3
if num_folds == 1:
    axes = [axes]

for idx, cm in enumerate(all_conf_matrices):
    ax = axes[idx]
    im = ax.imshow(cm)

    ax.set_title(participants[idx], fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(unique_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(unique_labels, fontsize=8)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=axes, shrink=0.8)
plt.show()

print("\nAverage Accuracy across LOPO:", np.mean(all_accuracies))
