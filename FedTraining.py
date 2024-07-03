import copy
import os
import time
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from CNN import CNN
from utils import load_dataset, visualize_dataset

NUM_EPOCHS = 10
LOCAL_ITERS = 2
VIS_DATA = False
BATCH_SIZE = 10
NUM_CLIENTS = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

#DATASET = "grayscale_mnist_3_channels"
DATASET = "color_mnist"



def FedAvg(params):
    global_params = copy.deepcopy(params[0])
    for key in global_params.keys():
        for param in params[1:]:
            global_params[key] += param[key]
        global_params[key] = torch.div(global_params[key], len(params))
    return global_params


def train(local_model, device, dataset, iters):
    local_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    train_loss = 0.0
    local_model.train()
    for i in range(iters):
        batch_loss = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(dataset), total=len(dataset)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            batch_loss += loss.item() * data.size(0)
            loss.backward()
            optimizer.step()
        train_loss += batch_loss / len(dataset)
    return local_model.state_dict(), train_loss / iters

def test(model, dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset)/BATCH_SIZE):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return test_loss / len(dataloader.dataset), accuracy

def check_data_distribution(dataloader, name):
    label_counts = np.zeros(10)
    for _, labels in dataloader:
        for label in labels:
            label_counts[label] += 1
    print(f"Data distribution in {name}: {label_counts}")

if __name__ == "__main__":
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')

    logname = (f"results/log_federated_{DATASET}_{str(NUM_EPOCHS)}_{str(NUM_CLIENTS)}_{str(LOCAL_ITERS)}")
    logging.basicConfig(filename=logname, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    train_data, validation_data, test_data = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset=DATASET)
    if VIS_DATA: visualize_dataset([train_data, validation_data, test_data])

    check_data_distribution(train_data, "Training Data")
    check_data_distribution(validation_data, "Validation Data")
    check_data_distribution(test_data, "Test Data")

    train_distributed_dataset = [[] for _ in range(NUM_CLIENTS)]
    for batch_idx, (data, target) in enumerate(train_data):
        train_distributed_dataset[batch_idx % NUM_CLIENTS].append((data, target))

    model_path = f"models/{DATASET}_{str(NUM_CLIENTS)}_federated.sav"

    all_train_loss = list()
    all_val_loss = list()
    all_val_acc = list()
    epoch_times = list()

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        global_model = CNN()
        global_model.load_state_dict(torch.load(model_path))
    else:
        global_model = CNN()
        global_params = global_model.state_dict()

        global_model.train()
        val_loss_min = np.Inf

        for epoch in range(1, NUM_EPOCHS+1):
            print(f"Epoch {str(epoch)}")
            local_params, local_losses = [], []

            start_time = time.time()
            for idx in range(NUM_CLIENTS):
                param, loss = train(copy.deepcopy(global_model), DEVICE, train_distributed_dataset[idx], LOCAL_ITERS)
                local_params.append(copy.deepcopy(param))
                local_losses.append(copy.deepcopy(loss))
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            global_params = FedAvg(local_params)
            global_model.load_state_dict(global_params)
            train_loss = sum(local_losses)/len(local_losses)
            all_train_loss.append(train_loss)

            val_loss, val_acc = test(global_model, validation_data)
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

            print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
            logger.info(f'Epoch: {epoch}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Val Accuracy: {val_acc:.8f}')

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                print("Saving model")
                torch.save(global_model.state_dict(), model_path)

        print("Testing with IID (Bias-Aligned) setting:")
        test_loss, test_acc = test(global_model, test_data)
        print(f"IID Test accuracy: {test_acc:.8f}")
        logger.info(f'IID Test Loss: {test_loss:.8f}, IID Test Accuracy: {test_acc:.8f}')

    """
    You must test on the three different datasets, dont you think? youre only testing on two, what if you trained the model with already one of these?
    """

    # Testing with Bias-Neutral (Grayscale) setting
    _, _, test_data_gray = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="grayscale_mnist_3_channels")
    print("Testing with Bias-Neutral (Grayscale) setting:")
    val_loss, val_acc = test(global_model, test_data_gray)
    print(f"Bias-Neutral Test Loss: {val_loss}, Bias-Neutral Test Accuracy: {val_acc}")
    logger.info(f'Bias-Neutral Test Loss: {val_loss:.8f}, Bias-Neutral Test Accuracy: {val_acc:.8f}')

    # Testing with Bias-Conflicting setting
    _, _, bias_conflicting_test = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="bias_conflicting_mnist")
    print("Testing with Bias-Conflicting setting:")
    test_loss, test_acc = test(global_model, bias_conflicting_test)
    print(f"Bias-Conflicting Test Loss: {test_loss}, Bias-Conflicting Test Accuracy: {test_acc}")
    logger.info(f'Bias-Conflicting Test Loss: {test_loss:.8f}, Bias-Conflicting Test Accuracy: {test_acc:.8f}')

    results = {
        "train_loss": all_train_loss,
        "val_loss": all_val_loss,
        "val_acc": all_val_acc,
        "epoch_times": epoch_times
    }
    np.save(f"results/results_{DATASET}_{NUM_CLIENTS}.npy", results)
