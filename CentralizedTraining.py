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

logging.getLogger('matplotlib').setLevel(logging.WARNING)

NUM_EPOCHS = 3
VIS_DATA = False
BATCH_SIZE = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)
DATASET = "grayscale_mnist_3_channels"
exp_num = 2
apply_l2 = False


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


def combine_datasets(color_data, gray_data, exp_num):
    num_color = len(color_data) // 5
    num_gray = len(gray_data) // 5
    color_subsets = [[] for _ in range(5)]
    gray_subsets = [[] for _ in range(5)]

    for batch_idx, (data, target) in enumerate(color_data):
        color_subsets[batch_idx % 5].append((data, target))
    for batch_idx, (data, target) in enumerate(gray_data):
        gray_subsets[batch_idx % 5].append((data, target))

    combined_data = []
    match exp_num:
        case 1:
            combined_data = color_subsets[0] + [item for subset in gray_subsets[1:] for item in subset]
        case 2:
            combined_data = [item for subset in color_subsets[:2] for item in subset] + [item for subset in gray_subsets[2:] for item in subset]
        case 3:
            combined_data = [item for subset in color_subsets[:3] for item in subset] + gray_subsets[3]
        case 4:
            combined_data = [item for subset in color_subsets for item in subset]
        case 5:
            combined_data = [item for subset in gray_subsets for item in subset]

    return combined_data


if __name__ == "__main__":
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')

    #logname = ('results/log_baseline_' + DATASET + "_" + str(NUM_EPOCHS))
    logname = f'results/log_centralized_exp{exp_num}_{NUM_EPOCHS}_{str(apply_l2)}.log'
    print(f"logname: {logname}")
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    logger = logging.getLogger()
    model_path = f"models/{exp_num}_{NUM_EPOCHS}_{str(apply_l2)}_centralized.sav"

    #train_data, validation_data, test_data = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset=DATASET)
    #if VIS_DATA: visualize_dataset([train_data, validation_data, test_data])


    train_color, val_color, test_color = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="color_mnist")
    train_gray3, val_gray3, test_gray3 = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="grayscale_mnist_3_channels")
    train_bias_conflict, val_bias_conflict, test_bias_conflict = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="bias_conflicting_mnist")
    """
    Should I combine the datasets too for the testing?
    Thoughts:
    - While it could be descriptive, I believe that maintaining the bias-aligned, bias-conflicting and bias neutral tests
        could be the most useful.
        Thus: Here use the gray3 mnist for the loss function
        And the color mnist for the final testing?
    """

    train_data = combine_datasets(train_color, train_gray3, exp_num)

    #validation_data = combine_datasets(val_color, val_gray3, exp_num)
    #test_data = combine_datasets(test_color, test_gray3, exp_num)


    #validation_data = val_gray3
    # let's try to change the validation data to bias conflicting, so that the loss function will use it
    validation_data = val_bias_conflict
    test_data = test_color # this one kinda not necessary because here I divided the bias aligned to be a different test (should do the same for federated...)


    # Dataloaders?
    #train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    #validation_data = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)


    model = CNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.Inf

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        start_time = time.time()
        total_data_transferred = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print("\nEpoch :", str(epoch))
            epoch_start_time = time.time()

            train_loss = train(model, DEVICE, train_data, criterion, optimizer, apply_l2=apply_l2, lambda_l2=0.01)
            val_loss, _, accuracy = test(model, validation_data, criterion)
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            print(
                f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {accuracy}")
            logger.info(
                'Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch, NUM_EPOCHS,
                                                                                                  train_loss,
                                                                                                  val_loss,
                                                                                                  accuracy))
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

    model.load_state_dict(torch.load(model_path))

    # Test on different datasets
    print("Testing with IID (Bias-Aligned) setting:")
    _, _, test_data_bias_aligned = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="color_mnist")
    _, _, accuracy_bias_aligned = test(model, test_data_bias_aligned, criterion)
    print(f"IID Test accuracy: {accuracy_bias_aligned:.8f}")
    logger.info('Bias-Aligned Test accuracy {:.8f}'.format(accuracy_bias_aligned))

    print("Testing with Bias-Neutral (Grayscale) setting:")
    _, _, test_data_grayscale = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="grayscale_mnist_3_channels")
    _, _, accuracy_grayscale = test(model, test_data_grayscale, criterion)
    logger.info('Bias-Neutral Test accuracy {:.8f}'.format(accuracy_grayscale))
    print(f"Bias-Neutral Test Accuracy: {accuracy_grayscale:.8f}")

    print("Testing with Bias-Conflicting setting:")
    _, _, test_data_bias_conflicting = load_dataset(val_split=0.2, batch_size=BATCH_SIZE,
                                                    dataset="bias_conflicting_mnist")
    _, _, accuracy_bias_conflicting = test(model, test_data_bias_conflicting, criterion)
    logger.info('Bias-Conflicting Test accuracy {:.8f}'.format(accuracy_bias_conflicting))
    print(f"Bias-Conflicting Test Accuracy: {accuracy_bias_conflicting:.8f}")