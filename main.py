import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import argparse
import matplotlib.pyplot as plt
import os

from data import load_dataset, preprocess_data
from model import GAT

def train_and_evaluate(graph, features, labels, train_mask, val_mask, test_mask, in_dim, hid_dim, out_dim, dropout, num_heads, epochs, lr, alpha, activation, weight_decay, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(in_dim, hid_dim, out_dim, dropout, alpha, num_heads, activation).to(device)
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    val_accs = []

    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        logits = model(features, graph)
        logits = logits.squeeze(1)

        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, graph).squeeze(1)
            val_loss = loss_fn(val_logits[val_mask], labels[val_mask])
            _, val_pred = torch.max(val_logits[val_mask], dim=1)
            val_acc = (val_pred == labels[val_mask]).float().mean().item()
            val_accs.append(val_acc)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Val Acc: {val_acc}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping")
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    model.eval()
    with torch.no_grad():
        test_logits = model(features, graph).squeeze(1)
        _, test_pred = torch.max(test_logits[test_mask], dim=1)
        test_acc = (test_pred == labels[test_mask]).float().mean().item()

    return model, val_accs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name (Cora, Citeseer, Pubmed)")
    parser.add_argument("--activation", type=str, default="elu", help="Activation function (elu, leaky_relu)")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    args = parser.parse_args()

    if args.activation == "elu":
        activation = F.elu
    elif args.activation == "leaky_relu":
        activation = F.leaky_relu
    else:
        raise ValueError(f"Unknown activation function: {args.activation}")

    dataset_name = args.dataset
    num_trials = args.trials

    print(f"Processing dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    g, features, labels, train_mask, val_mask, test_mask = preprocess_data(dataset)
    in_dim = features.shape[1]
    hid_dim = 8
    out_dim = len(torch.unique(labels))
    num_heads = 8
    epochs = 100
    dropout = 0.6
    alpha = 0.2
    patience = 10
    total_training_time = 0
    lr = 0.005

    # Weight decay 설정
    if dataset_name == "Pubmed":
        weight_decay = 0.001
    else:
        weight_decay = 0.0005

    all_val_accs = []
    all_test_accs = []
    for _ in range(num_trials):
        start_time = time.time()
        model, val_accs, test_acc = train_and_evaluate(
            g, features, labels, train_mask, val_mask, test_mask,
            in_dim, hid_dim, out_dim, dropout, num_heads, epochs, lr, alpha, activation, weight_decay, patience
        )
        end_time = time.time()
        training_time = end_time - start_time
        total_training_time += training_time
        all_val_accs.append(max(val_accs))
        all_test_accs.append(test_acc)
        print(f"Finished processing dataset: {dataset_name}")
        print(f"Best Validation Accuracy: {max(val_accs)}\n")
        print(f"Training time for this trial: {training_time:.2f} seconds\n")

    # accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(val_accs)+1), val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{dataset_name} - Validation Accuracy({activation})")
    os.makedirs("results", exist_ok=True)  # results 디렉토리 생성
    plt.savefig(f"results/{dataset_name}_val_acc_{activation}.png")
    plt.close()
    plt.show()

    avg_val_acc = sum(all_val_accs) / num_trials
    avg_training_time = total_training_time / num_trials
    print(f"Finished processing dataset: {dataset_name}")
    print(f"Average Best Validation Accuracy over {num_trials} trials: {avg_val_acc}\n")
    print(f"Average Training Time over {num_trials} trials: {avg_training_time:.2f} seconds\n")