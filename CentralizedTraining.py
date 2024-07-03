import os
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from CNN import CNN
import matplotlib.pyplot as plt
from utils import load_dataset, visualize_dataset

NUM_EPOCHS = 1
VIS_DATA = False
BATCH_SIZE = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)
DATASET = "grayscale_mnist_3_channels"


def train(model, device, dataloader, criterion, optimizer):
    train_loss = 0.0
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset) / BATCH_SIZE):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader.dataset)


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


if __name__ == "__main__":
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')

    logname = ('results/log_baseline_' + DATASET + "_" + str(NUM_EPOCHS))
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    logger = logging.getLogger()

    train_data, validation_data, test_data = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset=DATASET)
    if VIS_DATA: visualize_dataset([train_data, validation_data, test_data])

    model = CNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.Inf

    for epoch in range(1, NUM_EPOCHS + 1):
        print("\nEpoch :", str(epoch))
        train_loss = train(model, DEVICE, train_data, criterion, optimizer)
        val_loss, _, accuracy = test(model, validation_data, criterion)
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        logger.info('Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch, NUM_EPOCHS,
                                                                                                      train_loss,
                                                                                                      val_loss,
                                                                                                      accuracy))
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(model.state_dict(), "models/mnist_baseline.sav")

    model.load_state_dict(torch.load("models/mnist_baseline.sav"))
    test_loss, predictions, accuracy = test(model, test_data, criterion)
    logger.info('Test accuracy {:.8f}'.format(accuracy))

    # Test on Bias-Aligned
    _, _, test_data_bias_aligned = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="color_mnist")
    _, _, accuracy_bias_aligned = test(model, test_data_bias_aligned, criterion)
    logger.info('Bias-Aligned Test accuracy {:.8f}'.format(accuracy_bias_aligned))

    # Test on Bias-Neutral (Grayscale)
    _, _, test_data_grayscale = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset="grayscale_mnist_3_channels")
    _, _, accuracy_grayscale = test(model, test_data_grayscale, criterion)
    logger.info('Bias-Neutral Test accuracy {:.8f}'.format(accuracy_grayscale))

    # Test on Bias-Conflicting
    _, _, test_data_bias_conflicting = load_dataset(val_split=0.2, batch_size=BATCH_SIZE,
                                                    dataset="bias_conflicting_mnist")
    _, _, accuracy_bias_conflicting = test(model, test_data_bias_conflicting, criterion)
    logger.info('Bias-Conflicting Test accuracy {:.8f}'.format(accuracy_bias_conflicting))
