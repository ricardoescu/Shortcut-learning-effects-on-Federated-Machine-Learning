import torch
import copy
from tqdm import tqdm
apply_l2 = True
import os
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = 'SGD'

def train(local_model, device, dataset, test_dataset, iters, apply_l2=apply_l2, lambda_l2=0.01):
    """
    Trains a local model for a given client.
    :param local_model: A copy of global CNN model required for training
    :param device: The device to train the model on - GPU/CPU
    :param dataset: Training dataset
    :param iters:
    :return local_params: parameters from the trained model from the client
    :return train_loss: training loss for the current epoch
    :return local_accuracy: local accuracy after training
    """

    # Optimizer for training the local models
    local_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if opt == 'Adam':
        optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.AdamW(local_model.parameters(), lr=0.001)
    train_loss = 0.0
    local_model.train()

    # Iterate for the given number of client iterations
    for i in range(iters):
        batch_loss = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(dataset), total=len(dataset)):
            data, target = data.to(device), target.to(device)
            # Set gradients to zero
            optimizer.zero_grad()
            # Get output prediction from the Client model.
            output = local_model(data)
            # Compute loss
            loss = criterion(output, target)

            # manual L2
            if apply_l2:
                l2_reg = 0.0
                for name, param in local_model.named_parameters():
                    if 'weight' in name:
                        l2_reg += torch.norm(param, 2)
                loss += lambda_l2 * l2_reg

            batch_loss += loss.item() * data.size(0)
            # Collect new set of gradients
            loss.backward()
            # Update the local model
            optimizer.step()
        # add loss for each iteration
        train_loss += batch_loss / len(dataset)

    local_loss, local_accuracy = test(local_model, test_dataset)

    return local_model.state_dict(), train_loss / iters, local_accuracy


def test(model, dataset):
    """
    Tests the FL global model for the given dataset
    :param model: Trained CNN model for testing
    :param dataset: Dataset used to test the model
    :return test_loss: test loss for the current dataset
    :return accuracy: accuracy for the prediction values from the model
    """
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0  # Total number of samples
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(dataset):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
            total += target.size(0)  # Increment the total number of samples
    accuracy = correct / total  # Use total number of samples for accuracy calculation
    return test_loss / total, accuracy


def fedProx_train(local_model, global_model, device, dataset, test_dataset, iters, mu, apply_l2=apply_l2, lambda_l2=0.01):
    """
    Trains a local model for a given client with FedProx
    :param local_model: A copy of global CNN model required for training
    :param global_model: The global model to incorporate the proximal term
    :param device: The device to train the model on - GPU/CPU
    :param dataset: Training dataset
    :param iters:
    :param mu: Proximal term coeffiecient for the local model
    :return local_params: parameters from the trained model from the client
    :return train_loss: training loss for the current epoch
    :return local_accuracy: local accuracy after training
    """

    # Optimizer for training the local models
    local_model.to(device)
    global_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    train_loss = 0.0
    local_model.train()

    # Iterate for the given number of client iterations
    for i in range(iters):
        batch_loss = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(dataset), total=len(dataset)):
            data, target = data.to(device), target.to(device)
            # Set gradients to zero
            optimizer.zero_grad()
            # Get output prediction from the Client model.
            output = local_model(data)
            # Compute loss
            loss = criterion(output, target)

            """
            Test with proximal term
            """
            prox_term = 0.0
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                prox_term += torch.norm(local_param - global_param) ** 2
            loss += (mu / 2) * prox_term

            # manual L2
            if apply_l2:
                l2_reg = 0.0
                for name, param in local_model.named_parameters():
                    if 'weight' in name:
                        l2_reg += torch.norm(param, 2)
                loss += lambda_l2 * l2_reg

            batch_loss += loss.item() * data.size(0)
            # Collect new set of gradients
            loss.backward()
            # Update the local model
            optimizer.step()
        # add loss for each iteration
        train_loss += batch_loss / len(dataset)

    local_loss, local_accuracy = test(local_model, test_dataset)

    return local_model.state_dict(), train_loss / iters, local_accuracy