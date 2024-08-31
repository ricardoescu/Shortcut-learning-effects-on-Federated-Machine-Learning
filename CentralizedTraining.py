import os
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from CNN import CNN
import matplotlib.pyplot as plt
from utils import load_dataset, visualize_dataset
from torch.utils.data import DataLoader, ConcatDataset, Subset
import psutil
import time
import random

logging.getLogger('matplotlib').setLevel(logging.WARNING)

NUM_EPOCHS = 10
VIS_DATA = False
BATCH_SIZE = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)
DATASET = "grayscale_mnist_3_channels"
exp_num = 1
if exp_num == 9:
    NUM_CLIENTS = 3
else:
    NUM_CLIENTS = 4
apply_l2 = False
sub_exp_num = 1
opt = 'SGD'


def train(model, device, dataset, criterion, optimizer, apply_l2=False, lambda_l2=0.01):
    train_loss = 0.0
    model.train()
    #for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset) / BATCH_SIZE):
    for batch_idx, (data, target) in tqdm(enumerate(dataset), total=len(dataset) / BATCH_SIZE):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if apply_l2:
            l2_reg = 0.0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_reg += torch.norm(param, 2)
            loss += lambda_l2 * l2_reg

        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

    return train_loss / len(dataset)


def test(model, dataloader, criterion):
    test_loss = 0.0
    correct = 0
    model.eval()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset) / BATCH_SIZE):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(target.view_as(preds)).sum().item()
    accuracy = correct / len(dataloader.dataset)

    return test_loss / len(dataloader.dataset), preds, accuracy

def combine_datasets(cMNIST_A, gray_data, cMNIST_B, exp_num):
    num_color = len(cMNIST_A) // NUM_CLIENTS
    num_gray = len(gray_data) // NUM_CLIENTS
    cMNIST_A_Subsets = [[] for _ in range(NUM_CLIENTS)]
    gray_subsets = [[] for _ in range(NUM_CLIENTS)]
    cMNIST_B_Subsets = [[] for _ in range(NUM_CLIENTS)]



    for batch_idx, (data, target) in enumerate(cMNIST_A):
        cMNIST_A_Subsets[batch_idx % NUM_CLIENTS].append((data, target))
    for batch_idx, (data, target) in enumerate(gray_data):
        gray_subsets[batch_idx % NUM_CLIENTS].append((data, target))
    for batch_idx, (data, target) in enumerate(cMNIST_B):
        cMNIST_B_Subsets[batch_idx % NUM_CLIENTS].append((data, target))

    random.shuffle(cMNIST_A_Subsets)
    random.shuffle(gray_subsets)
    random.shuffle(cMNIST_B_Subsets)

    combined_data = []
    match exp_num:
        case 1:
            combined_data = cMNIST_A_Subsets[0] + [item for subset in gray_subsets[1:] for item in subset]
        case 2:
            combined_data = [item for subset in cMNIST_A_Subsets[:2] for item in subset] + [item for subset in gray_subsets[2:] for item in subset]
        case 3:
            combined_data = [item for subset in cMNIST_A_Subsets[:3] for item in subset] + gray_subsets[3]
        case 4:
            combined_data = [item for subset in cMNIST_A_Subsets for item in subset]
        case 5:
            combined_data = [item for subset in gray_subsets for item in subset]
        case 6:
            # 25% 1 get Color MNIST A, 25% get Color MNIST B, 50% get gray MNIST
            combined_data = cMNIST_A_Subsets[0] + cMNIST_B_Subsets[1] + [item for subset in gray_subsets[2:] for item in subset]
        case 7:
            # Client 1 gets grayscale mnist, Client 2 gets Color MNIST B, Clients 3 and 4 get color MNIST A
            combined_data = ([item for subset in cMNIST_A_Subsets[:2] for item in subset] + cMNIST_B_Subsets[1] +
                             gray_subsets[2])
        case 8:
            # First half get CMNISTA data, second half get CMNISTB data
            combined_data = ([item for subset in cMNIST_A_Subsets[:2] for item in subset] +
                             [item for subset in cMNIST_B_Subsets[2:] for item in subset])
        case 9:
            combined_data = cMNIST_A_Subsets[0] + cMNIST_B_Subsets[1] + gray_subsets[2]


    return combined_data
def setup_directories():
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')

def setup_logging(exp_num, NUM_EPOCHS, apply_l2):
    logname = f'results/log_centralized_exp{exp_num}_{NUM_EPOCHS}_{str(apply_l2)}_{sub_exp_num}_{opt}.log'
    print(f"logname: {logname}")
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    logger = logging.getLogger()
    return logger

def load_data(VIS_DATA, BATCH_SIZE):
    train_color, val_color, test_color = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="color_mnist")
    train_gray3, val_gray3, test_gray3 = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="grayscale_mnist_3_channels")
    train_bias_conflict, val_bias_conflict, test_bias_conflict = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="color_MNIST_B")

    if VIS_DATA:
        visualize_dataset([train_color, val_color, test_color])
        visualize_dataset([train_gray3, val_gray3, test_gray3])

    return train_color, val_color, test_color, train_gray3, val_gray3, test_gray3, train_bias_conflict, val_bias_conflict, test_bias_conflict


def initialize_model(model_path):
    model = CNN().to(DEVICE)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        model.train()
    return model


def train_loop(model, train_data, validation_data, val_MNIST_A, val_MNIST_B, criterion, optimizer, model_path, logger):
    all_train_loss = []
    all_val_loss = []
    val_loss_min = np.Inf
    start_time = time.time()
    total_data_transferred = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print("\nEpoch :", str(epoch))
        epoch_start_time = time.time()

        train_loss = train(model, DEVICE, train_data, criterion, optimizer, apply_l2=apply_l2, lambda_l2=0.01)
        val_loss, _, accuracy = test(model, validation_data, criterion)
        _, _, acc_cMNIST_A = test(model, val_MNIST_A, criterion)
        _, _, acc_cMNIST_B = test(model, val_MNIST_B, criterion)
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        print(
            f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {accuracy}, Bias aligned accuracy: {acc_cMNIST_A}, Bias conflicting accuracy: {acc_cMNIST_B}")
        logger.info(
            f'Epoch: {epoch}/{NUM_EPOCHS}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Val Accuracy: {accuracy:.8f}, Bias aligned accuracy: {acc_cMNIST_A}, Bias conflicting accuracy: {acc_cMNIST_B}')

        epoch_time = time.time() - epoch_start_time
        logger.info(f'Epoch {epoch} training time: {epoch_time:.2f} seconds')
        total_data_transferred += sum([param.element_size() * param.nelement() for param in model.parameters()])

        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(model.state_dict(), model_path)

    total_training_time = time.time() - start_time
    avg_cpu_usage = np.mean([psutil.cpu_percent(interval=1) for _ in range(10)])
    avg_memory_usage = np.mean([psutil.virtual_memory().percent for _ in range(10)])
    logger.info(f'Total training time: {total_training_time:.2f} seconds')
    logger.info(f'Total data transferred: {total_data_transferred / (1024 ** 2):.2f} MB')
    logger.info(f'Average CPU usage: {avg_cpu_usage:.2f}%')
    logger.info(f'Average memory usage: {avg_memory_usage:.2f}%')

    return model


def evaluate_model(model, criterion, logger):
    _, _, accuracy_bias_aligned = test(model, test_color, criterion)
    print(f"IID Test accuracy: {accuracy_bias_aligned:.8f}")
    logger.info('Bias-Aligned Test accuracy {:.8f}'.format(accuracy_bias_aligned))

    _, _, accuracy_grayscale = test(model, test_gray3, criterion)
    logger.info('Bias-Neutral Test accuracy {:.8f}'.format(accuracy_grayscale))
    print(f"Bias-Neutral Test Accuracy: {accuracy_grayscale:.8f}")

    _, _, accuracy_bias_conflicting = test(model, test_bias_conflict, criterion)
    logger.info('Bias-Conflicting Test accuracy {:.8f}'.format(accuracy_bias_conflicting))
    print(f"Bias-Conflicting Test Accuracy: {accuracy_bias_conflicting:.8f}")

    train_MNIST_C, val_MNIST_C, test_MNIST_C = load_dataset(val_split=0.2, batch_size=BATCH_SIZE,
                                                            dataset="color_MNIST_C")
    _, _, accuracy_cMNIST_C = test(model, test_MNIST_C, criterion)
    logger.info('cMNIST C Test accuracy {:.8f}'.format(accuracy_cMNIST_C))
    print(f"cMNIST C Test Accuracy: {accuracy_cMNIST_C:.8f}")

if __name__ == "__main__":
    setup_directories()
    logger = setup_logging(exp_num, NUM_EPOCHS, apply_l2)

    train_color, val_color, test_color, train_gray3, val_gray3, test_gray3, train_bias_conflict, val_bias_conflict, test_bias_conflict = load_data(
        VIS_DATA, BATCH_SIZE)
    train_MNIST_C, val_MNIST_C, test_MNIST_C = load_dataset(val_split=0.2, batch_size=BATCH_SIZE,
                                                            dataset="color_MNIST_C")



    """
    Should I combine the datasets too for the testing?
    Thoughts:
    - While it could be descriptive, I believe that maintaining the bias-aligned, bias-conflicting and bias neutral tests
        could be the most useful.
        Thus: Here use the gray3 mnist for the loss function
        And the color mnist for the final testing?
    """

    train_data = combine_datasets(train_color, train_gray3, train_bias_conflict, exp_num)


    validation_data = val_gray3
    # let's try to change the validation data to bias conflicting, so that the loss function will use it
    #validation_data = val_bias_conflict
    test_data = test_color # this one kinda not necessary because here I divided the bias aligned to be a different test (should do the same for federated...)


    model_path = f"models/{exp_num}_{NUM_EPOCHS}_{str(apply_l2)}_{opt}_{sub_exp_num}_centralized.sav"

    model = initialize_model(model_path)
    criterion = torch.nn.CrossEntropyLoss()
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    if model.training:
        model = train_loop(model, train_data, validation_data, val_color, val_bias_conflict, criterion, optimizer, model_path, logger)

    evaluate_model(model, criterion, logger)
