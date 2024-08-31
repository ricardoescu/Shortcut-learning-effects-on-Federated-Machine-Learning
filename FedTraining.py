import os
import time
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from CNN import CNN
import random
from utils import load_dataset, visualize_dataset, check_data_distribution, visualize_client_data
import copy
from torch.utils.data import random_split, Subset
import matplotlib.pyplot as plt
import psutil
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from Fed_strategies import FedAvg
from Fed_train_test import train, test, fedProx_train


NUM_EPOCHS = 10
LOCAL_ITERS = 1
VIS_DATA = False
BATCH_SIZE = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
exp_num = 1
if exp_num == 9:
    NUM_CLIENTS = 3
else:
    NUM_CLIENTS = 4
apply_l2 = False
#algorithm = 'FedProx'
algorithm = 'FedAvg'
sub_exp_num = 1
#mu = 0.01
#mu = 1
opt = 'SGD'


def setup_directories():
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')


def setup_logging(exp_num, NUM_EPOCHS, NUM_CLIENTS, LOCAL_ITERS, apply_l2):
    # Initialize a logger to log epoch results
    # logname = (f"results/log_federated_{DATASET}_{str(NUM_EPOCHS)}_{str(NUM_CLIENTS)}_{str(LOCAL_ITERS)}")

    root_logger = logging.getLogger()

    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logname = (
        f"results/log_federated_Exp-{exp_num}_Epochs-{str(NUM_EPOCHS)}_Clients-{str(NUM_CLIENTS)}_L2-{str(apply_l2)}_strat-{algorithm}_{sub_exp_num}_{opt}.log")
    print(f"logname: {logname}")
    logging.basicConfig(filename=logname, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger


def load_and_visualize_data(VIS_DATA, BATCH_SIZE):
    # Get data for color mnist
    # train_data, validation_data, test_data = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset=DATASET)
    train_color, validation_color, test_color = load_dataset(val_split=0.2, batch_size=BATCH_SIZE,
                                                             dataset="color_mnist")

    # Get data for grayscale MNIST
    train_gray3, validation_gray3, test_gray3 = load_dataset(val_split=0.2, batch_size=BATCH_SIZE,
                                                             dataset="grayscale_mnist_3_channels")
    if VIS_DATA:
        visualize_dataset([train_color, validation_color, test_color])
        visualize_dataset([train_gray3, validation_gray3, test_gray3])
    return train_color, validation_color, test_color, train_gray3, validation_gray3, test_gray3


def check_all_data_distributions(train_color, validation_color, test_color, train_gray3, validation_gray3, test_gray3):
    check_data_distribution(train_color, "Color Training Data")
    check_data_distribution(validation_color, "Color Validation Data")
    check_data_distribution(test_color, "Color Test Data")

    check_data_distribution(train_gray3, "Gray3 Training Data")
    check_data_distribution(validation_gray3, "Gray3 Validation Data")
    check_data_distribution(test_gray3, "Gray3 Test Data")


# Function to distribute data for different experiment settings
def distribute_data(num_clients, cMNIST_A, gray_data, cMNIST_B, exp_num):
    """
    This function arranges the data in 5 different distributions depending on the experiment to perform.
    Experiment 1: Only client 1 has color mnist. Clients 2, 3, and 4 have gray mnist
    Experiment 2: Clients 1, 2 have color mnist. Clients 3, 4 have gray MNIST
    Experiment 3: Client 1, 2, 3 have color MNIST. Client 4 has only gray MNIST.
    Experiment 4: All clients have only color MNIST.
    Experiment 5: All clients have only grayscale MNIST.
    Experiment 6: Client 1 gets cMNIST A, client 2 gets cMNIST B, clients 3 and 4 get gray MNIST.
    Experiment 7: Client 1 gets grayscale MNIST, client gets cMNIST B, clients 3 and 4 get cMNIST A
    Experiment 8: Clients 1 and 2 get CMNIST A, clients 3 and 4 get cMNIST B
    Experiment 9: Client 1 gets cMNIST A, client 2 gets cMNIST B, client 3 gets MNIST.
    :param num_clients:
    :param cMNIST_A:
    :param gray_data: Gray3 (Grayscale MNIST adjusted to have 3 color channels so there are no problems with the CNN model)
    :param exp_num: Experiment number.
    :return:
    """
    distributed_data = [[] for _ in range(num_clients)]


    # Splitting color and gray data into equal parts for clients
    color_subsets_A = [[] for _ in range(num_clients)]
    gray_subsets = [[] for _ in range(num_clients)]
    color_subsets_B = [[] for _ in range(num_clients)]


    for batch_idx, (data, target) in enumerate(cMNIST_A):
        color_subsets_A[batch_idx % num_clients].append((data, target))
    for batch_idx, (data, target) in enumerate(gray_data):
        gray_subsets[batch_idx % num_clients].append((data, target))
    for batch_idx, (data, target) in enumerate(cMNIST_B):
        color_subsets_B[batch_idx % num_clients].append((data, target))

    # shuffle the datasets so that i can get an average??
    random.shuffle(color_subsets_A)
    random.shuffle(gray_subsets)
    random.shuffle(color_subsets_B)

    if exp_num == 1:
        # Client 1 gets 1/num_clients of color data, others get gray data
        distributed_data[0] = color_subsets_A[0]
        for i in range(1, num_clients):
            distributed_data[i] = gray_subsets[i]
    elif exp_num == 2:
        # First half clients get gray data, second half get color data
        for i in range(num_clients // 2):
            distributed_data[i] = gray_subsets[i]
        for i in range(num_clients // 2, num_clients):
            distributed_data[i] = color_subsets_A[i - num_clients // 2]
    elif exp_num == 3:
        # All but the last client get color data, the last client gets gray data
        for i in range(num_clients - 1):
            distributed_data[i] = color_subsets_A[i]
        distributed_data[num_clients - 1] = gray_subsets[num_clients - 1]
    elif exp_num == 4:
        # All clients get color data
        for i in range(num_clients):
            distributed_data[i] = color_subsets_A[i]
    elif exp_num == 5:
        # All clients get gray data
        for i in range(num_clients):
            distributed_data[i] = gray_subsets[i]
    elif exp_num == 6:
        # Client 1 gets Color MNIST A, Client 2 gets Color MNIST B, Clients 3 and 4 get gray MNIST
        distributed_data[0] = color_subsets_A[0]
        distributed_data[1] = color_subsets_B[0]
        for i in range(2, num_clients):
            distributed_data[i] = gray_subsets[i]
    elif exp_num == 7:
        # Client 1 gets grayscale mnist, Client 2 gets Color MNIST B, Clients 3 and 4 get color MNIST A
        distributed_data[0] = gray_subsets[0]
        distributed_data[1] = color_subsets_B[0]
        for i in range(2, num_clients):
            distributed_data[i] = color_subsets_A[i]
    elif exp_num == 8:
        # First half clients get CMNISTA data, second half get CMNISTB data
        for i in range(num_clients // 2):
            distributed_data[i] = color_subsets_A[i]
        for i in range(num_clients // 2, num_clients):
            distributed_data[i] = color_subsets_B[i - num_clients // 2]
    elif exp_num == 9:
        # 1 client gets CMNISTA data, second client gets CMNISTB data, third gets grayscale MNIST
        distributed_data[0] = color_subsets_A[0]
        distributed_data[1] = color_subsets_B[1]
        distributed_data[2] = gray_subsets[2]

    for i, subset in enumerate(distributed_data):
        print(f"Client {i + 1} dataset size: {len(subset)}")

    return distributed_data


# Function to split dataset into subsets for each client
def split_dataset(dataset, num_clients):
    subsets = [[] for _ in range(num_clients)]
    for batch_idx, (data, target) in enumerate(dataset):
        subsets[batch_idx % num_clients].append((data, target))
    return subsets



def initialize_global_model(model_path):
    global_model = CNN()
    global_params = global_model.state_dict()
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        global_model.load_state_dict(torch.load(model_path))
    else:
        global_model.train()
    return global_model


def train_global_model(global_model, training_set, validation_set, test_set, test_CA, test_CB, NUM_EPOCHS, LOCAL_ITERS, NUM_CLIENTS, DEVICE, apply_l2, model_path, logger, algorithm='FedAvg', mu=0.01):
    global_params = global_model.state_dict()
    val_loss_min = np.Inf
    all_train_loss, all_val_loss, all_val_acc, epoch_times, all_local_acc = [], [], [], [], []
    total_data_transferred = 0

    # Train the model for the given number of epochs
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {str(epoch)}")
        local_params, local_losses = [], []
        local_accuracies = []
        epoch_start_time = time.time()

        # Send a copy of global model to each client
        for idx in range(NUM_CLIENTS):
            if algorithm == 'FedAvg':
                # Perform training on client side and get the parameters
                param, loss, local_accuracy = train(copy.deepcopy(global_model), DEVICE, training_set[idx], validation_set[idx],
                                                    LOCAL_ITERS)
            elif algorithm == 'FedProx':
                param, loss, local_accuracy = fedProx_train(copy.deepcopy(global_model), DEVICE, training_set[idx], validation_set[idx], LOCAL_ITERS, mu)

            else: raise ValueError(f"Unknown algorithm {algorithm}")

            local_params.append(copy.deepcopy(param))
            local_losses.append(copy.deepcopy(loss))
            local_accuracies.append(local_accuracy)
            logger.info(f'Client {idx + 1}, Local Training Accuracy: {local_accuracy:.8f}')
            print(f"local accuracy: {local_accuracy}")
            total_data_transferred += sum([param[key].element_size() * param[key].nelement() for key in param])
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        print(f'Epoch {epoch} time: {epoch_time:.2f} seconds')
        logger.info(f'Epoch {epoch} time: {epoch_time:.2f} seconds')
        all_local_acc.append(sum(local_accuracies) / len(local_accuracies))

        # Federated Average for the parameters from each client
        global_params = FedAvg(local_params)
        # Update the global model
        global_model.load_state_dict(global_params)
        train_loss = sum(local_losses) / len(local_losses)
        all_train_loss.append(train_loss)

        # Test the global model on the gray3 to ensure it tries to avoid the bias.
        val_loss, val_acc = test(global_model, test_set)
        # let's see how does the accuracy for bias-aligned evolves, also for bias-conflicting
        _, CMNISTA_acc= test(global_model, test_CA)
        _, CMNISTB_acc = test(global_model, test_CB)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)

        print(
            f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}, Bias aligned accuracy: {CMNISTA_acc}, Bias conflicting accuracy: {CMNISTB_acc}")
        logger.info(
            f'Epoch: {epoch}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Val Accuracy: {val_acc:.8f}, Bias aligned accuracy: {CMNISTA_acc:.4f}, Bias conflicting accuracy: {CMNISTB_acc:.4f}')

        # If validation loss decreases, save the model
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            print("Saving model")
            torch.save(global_model.state_dict(), model_path)

    return global_model, all_train_loss, all_val_loss, all_val_acc, epoch_times, all_local_acc, total_data_transferred


def evaluate_model(global_model, test_color, test_gray, bias_conflicting_test, mnistC, logger):
    test_loss, test_acc = test(global_model, test_color)
    print(f"IID Test accuracy: {test_acc:.8f}")
    logger.info(f'IID Test Loss: {test_loss:.8f}, IID Test Accuracy: {test_acc:.8f}')

    val_loss, val_acc = test(global_model, test_gray)
    print(f"Bias-Neutral Test Loss: {val_loss}, Bias-Neutral Test Accuracy: {val_acc}")
    logger.info(f'Bias-Neutral Test Loss: {val_loss:.8f}, Bias-Neutral Test Accuracy: {val_acc:.8f}')

    test_loss, test_acc = test(global_model, bias_conflicting_test)
    print(f"Bias-Conflicting Test Loss: {test_loss}, Bias-Conflicting Test Accuracy: {test_acc}")
    logger.info(f'Bias-Conflicting Test Loss: {test_loss:.8f}, Bias-Conflicting Test Accuracy: {test_acc:.8f}')

    test_loss, test_acc = test(global_model, mnistC)
    print(f"C MNIST_C Test Loss: {test_loss}, C MNIST_C Test accuracy: {test_acc}")
    logger.info(f'C MNIST_C Test Loss: {test_loss:.8f}, C MNIST_C Test Accuracy: {test_acc:.8f}')

def log_and_save_results(start_time, total_data_transferred, all_train_loss, all_val_loss, all_val_acc, all_local_acc, epoch_times, exp_num, NUM_CLIENTS, logger):
    total_training_time = time.time() - start_time
    avg_cpu_usage = np.mean([psutil.cpu_percent(interval=1) for _ in range(10)])
    avg_memory_usage = np.mean([psutil.virtual_memory().percent for _ in range(10)])
    logger.info(f'Total training time: {total_training_time:.2f} seconds')
    logger.info(f'Total data transferred: {total_data_transferred / (1024 ** 2):.2f} MB')
    logger.info(f'Average CPU usage: {avg_cpu_usage:.2f}%')
    logger.info(f'Average memory usage: {avg_memory_usage:.2f}%')
    results = {
        "train_loss": all_train_loss,
        "val_loss": all_val_loss,
        "val_acc": all_val_acc,
        "local_acc": all_local_acc,
        "epoch_times": epoch_times,
        "total_training_time": total_training_time,
        "total_data_transferred": total_data_transferred,
        "avg_cpu_usage": avg_cpu_usage,
        "avg_memory_usage": avg_memory_usage
    }
    print(f"results: {results}")
    np.save(f"results/results_{exp_num}_{NUM_CLIENTS}.npy", results)

if __name__ == "__main__":
    for i in range(1, 10):
        exp_num = i
        if exp_num == 9:
            NUM_CLIENTS = 3
        else:
            NUM_CLIENTS = 4
        for j in range(1, 4):
            sub_exp_num = j
            setup_directories()
            logger = setup_logging(exp_num, NUM_EPOCHS, NUM_CLIENTS, LOCAL_ITERS, apply_l2)
            start_time = time.time()

            train_color, validation_color, test_color, train_gray3, validation_gray3, test_gray3 = load_and_visualize_data(
                VIS_DATA, BATCH_SIZE)
            check_all_data_distributions(train_color, validation_color, test_color, train_gray3, validation_gray3, test_gray3)

            train_MNIST_B, val_MNIST_B, test_MNIST_B = load_dataset(val_split=0.2, batch_size=BATCH_SIZE,
                                                                    dataset="color_MNIST_B")

            train_MNIST_C, val_MNIST_C, test_MNIST_C = load_dataset(val_split=0.2, batch_size=BATCH_SIZE,
                                                                    dataset="color_MNIST_C")

            # Distribute the training data divided by color across clients
            #train_distributed_dataset = [[] for _ in range(NUM_CLIENTS)]
            train_distributed_dataset = distribute_data(NUM_CLIENTS, train_color, train_gray3, train_MNIST_B, exp_num)

            validation_gray3_splits = split_dataset(validation_gray3, NUM_CLIENTS)

            # Visualize one example from each client's dataset to ensure correct distribution.
            visualize_client_data(train_distributed_dataset)

            model_path = f"models/exp_num-{exp_num}_clients-{str(NUM_CLIENTS)}_L2-{str(apply_l2)}_strat-{algorithm}_{opt}_{sub_exp_num}_federated.sav"

            # Ensure that there is no need to train a model that has been previously trained under the same settings.
            global_model = initialize_global_model(model_path)

            if global_model.training:
                global_model, all_train_loss, all_val_loss, all_val_acc, epoch_times, all_local_acc, total_data_transferred = train_global_model(
                    global_model, train_distributed_dataset, validation_gray3_splits, test_gray3, test_color, test_MNIST_B, NUM_EPOCHS, LOCAL_ITERS, NUM_CLIENTS, DEVICE,
                    apply_l2, model_path, logger
                )


            evaluate_model(global_model, test_color, test_gray3, test_MNIST_B, test_MNIST_C, logger)
            log_and_save_results(start_time, total_data_transferred, all_train_loss, all_val_loss, all_val_acc, all_local_acc, epoch_times, exp_num, NUM_CLIENTS, logger)


