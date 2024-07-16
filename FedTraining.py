import os
import time
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from CNN import CNN
from utils import load_dataset, visualize_dataset
import copy
from torch.utils.data import random_split, Subset
import matplotlib.pyplot as plt
import psutil
logging.getLogger('matplotlib').setLevel(logging.WARNING)


NUM_EPOCHS = 3
LOCAL_ITERS = 1
VIS_DATA = False
BATCH_SIZE = 10
NUM_CLIENTS = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
exp_num = 2
apply_l2 = True


def FedAvg(params):
    """
    Average the parameters from each client to update the global model
    :param params: list of parameters from each client's model
    :return global_params: average of parameters from each client
    """
    global_params = copy.deepcopy(params[0])
    for key in global_params.keys():
        for param in params[1:]:
            global_params[key] += param[key]
        global_params[key] = torch.div(global_params[key], len(params))
    return global_params


def train(local_model, device, dataset, iters, apply_l2=apply_l2, lambda_l2=0.01):
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
    #optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001, weight_decay=0.01 if apply_l2 else 0.0)
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

    local_loss, local_accuracy = test(local_model, dataset)

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


def check_data_distribution(dataloader, name):
    label_counts = np.zeros(10)
    for _, labels in dataloader:
        for label in labels:
            label_counts[label] += 1
    #print(f"Data distribution in {name}: {label_counts}")


# Function to distribute data for different experiment settings
def distribute_data(num_clients, color_data, gray_data, exp_num):
    """
    This function arranges the data in 5 different distributions depending on the experiment to perform.
    Experiment 1: Only client 1 has color mnist. Clients 2, 3, and 4 have gray mnist
    Experiment 2: Clients 1, 2 have color mnist. Clients 3, 4 have gray MNIST
    Experiment 3: Client 1, 2, 3 have color MNIST. Client 4 has only gray MNIST.
    Experiment 4: All clients have only color MNIST.
    Experiment 5: All clients have only grayscale MNIST.
    :param num_clients:
    :param color_data:
    :param gray_data: Gray3 (Grayscale MNIST adjusted to have 3 color channels so there are no problems with the CNN model)
    :param exp_num: Experiment number.
    :return:
    """
    distributed_data = [[] for _ in range(num_clients)]

    # Splitting color and gray data into equal parts for clients
    color_subsets = [[] for _ in range(num_clients)]
    gray_subsets = [[] for _ in range(num_clients)]

    for batch_idx, (data, target) in enumerate(color_data):
        color_subsets[batch_idx % num_clients].append((data, target))
    for batch_idx, (data, target) in enumerate(gray_data):
        gray_subsets[batch_idx % num_clients].append((data, target))

    if exp_num == 1:
        # Client 1 gets 1/num_clients of color data, others get gray data
        distributed_data[0] = color_subsets[0]
        for i in range(1, num_clients):
            distributed_data[i] = gray_subsets[i]
    elif exp_num == 2:
        # First half clients get gray data, second half get color data
        for i in range(num_clients // 2):
            distributed_data[i] = gray_subsets[i]
        for i in range(num_clients // 2, num_clients):
            distributed_data[i] = color_subsets[i - num_clients // 2]
    elif exp_num == 3:
        # All but the last client get color data, the last client gets gray data
        for i in range(num_clients - 1):
            distributed_data[i] = color_subsets[i]
        distributed_data[num_clients - 1] = gray_subsets[num_clients - 1]
    elif exp_num == 4:
        # All clients get color data
        for i in range(num_clients):
            distributed_data[i] = color_subsets[i]
    elif exp_num == 5:
        # All clients get gray data
        for i in range(num_clients):
            distributed_data[i] = gray_subsets[i]

    for i, subset in enumerate(distributed_data):
        print(f"Client {i + 1} dataset size: {len(subset)}")

    return distributed_data


# Function to split dataset into subsets for each client
def split_dataset(dataset, num_clients):
    subset_size = len(dataset) // num_clients
    lengths = [subset_size] * num_clients
    lengths[-1] += len(dataset) % num_clients
    subsets = random_split(dataset, lengths)
    return subsets


def visualize_client_data(distributed_data):
    """
    Visualizes one example from each client's dataset.
    :param distributed_data: List of datasets distributed to each client.
    """
    num_clients = len(distributed_data)
    fig, axes = plt.subplots(1, num_clients, figsize=(15, 5))

    for i, client_data in enumerate(distributed_data):
        if len(client_data) > 0:
            # Get one sample
            data, target = client_data[0]

            # Remove batch dimension
            if len(data.shape) == 4:  # If there is a batch dimension
                data = data[0]
                target = target[0]

            # Handle grayscale and color images
            if data.shape[0] == 3:  # If the data is color (3 channels)
                data = data.permute(1, 2, 0)  # Change dimensions for plotting
                axes[i].imshow(data)
            else:  # Grayscale
                axes[i].imshow(data.squeeze(), cmap='gray')

            axes[i].set_title(f'Client {i + 1}, Label: {target.item()}')
            axes[i].axis('off')

    plt.show()


if __name__ == "__main__":
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')

    # Initialize a logger to log epoch results
    #logname = (f"results/log_federated_{DATASET}_{str(NUM_EPOCHS)}_{str(NUM_CLIENTS)}_{str(LOCAL_ITERS)}")
    logname = (f"results/log_federated_{exp_num}_{str(NUM_EPOCHS)}_{str(NUM_CLIENTS)}_{str(LOCAL_ITERS)}_{str(apply_l2)}.log")
    print(f"logname: {logname}")
    logging.basicConfig(filename=logname, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Get data for color mnist
    #train_data, validation_data, test_data = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset=DATASET)
    train_color, validation_color, test_color = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="color_mnist")
    if VIS_DATA: visualize_dataset([train_color, validation_color, test_color])

    # Get data for grayscale MNIST
    train_gray3, validation_gray3, test_gray3 = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="grayscale_mnist_3_channels")
    if VIS_DATA: visualize_dataset([train_gray3, validation_gray3, test_gray3])

    check_data_distribution(train_color, "Color Training Data")
    check_data_distribution(validation_color, "Color Validation Data")
    check_data_distribution(test_color, "Color Test Data")


    check_data_distribution(train_gray3, "Gray3 Training Data")
    check_data_distribution(validation_gray3, "Gray3 Validation Data")
    check_data_distribution(test_gray3, "Gray3 Test Data")

    # Distribute the training data divided by color across clients
    #train_distributed_dataset = [[] for _ in range(NUM_CLIENTS)]
    train_distributed_dataset = distribute_data(NUM_CLIENTS, train_color, train_gray3, exp_num)

    # Visualize one example from each client's dataset to ensure correct distribution.
    visualize_client_data(train_distributed_dataset)

    """for batch_idx, (data, target) in enumerate(train_data):
        train_distributed_dataset[batch_idx % NUM_CLIENTS].append((data, target))"""

    #model_path = f"models/{DATASET}_{str(NUM_CLIENTS)}_federated.sav"
    model_path = f"models/{exp_num}_{str(NUM_CLIENTS)}_{str(apply_l2)}_federated.sav"

    all_train_loss = list()
    all_val_loss = list()
    all_val_acc = list()
    epoch_times = list()

    all_local_acc = list()
    start_time = time.time()
    total_data_transferred = 0

    # Ensure that there is no need to train a model that has been previously trained under the same settings.
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        global_model = CNN()
        # Load best model from training
        global_model.load_state_dict(torch.load(model_path))
    else:
        # Get model and define criterion for loss
        global_model = CNN()
        global_params = global_model.state_dict()

        global_model.train()
        val_loss_min = np.Inf


        # Train the model for the given number of epochs
        for epoch in range(1, NUM_EPOCHS+1):
            print(f"Epoch {str(epoch)}")
            local_params, local_losses = [], []
            local_accuracies = []
            epoch_start_time = time.time()

            # Send a copy of global model to each client
            for idx in range(NUM_CLIENTS):
                # Perform training on client side and get the parameters
                param, loss, local_accuracy = train(copy.deepcopy(global_model), DEVICE, train_distributed_dataset[idx], LOCAL_ITERS)
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
            train_loss = sum(local_losses)/len(local_losses)
            all_train_loss.append(train_loss)

            # Test the global model on the gray3 to ensure it tries to avoid the bias.
            val_loss, val_acc = test(global_model, validation_gray3)
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

            print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
            logger.info(f'Epoch: {epoch}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Val Accuracy: {val_acc:.8f}')

            # If validation loss decreases, save the model
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                print("Saving model")
                torch.save(global_model.state_dict(), model_path)

        print("Testing with IID (Bias-Aligned) setting:")
        test_loss, test_acc = test(global_model, test_color)
        print(f"IID Test accuracy: {test_acc:.8f}")
        logger.info(f'IID Test Loss: {test_loss:.8f}, IID Test Accuracy: {test_acc:.8f}')

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
