import matplotlib.pyplot as plt
import numpy as np


def plot_global_accuracy(epochs, global_val_acc, title='Global Model Accuracy', save_path=None):
    plt.figure()
    plt.plot(epochs, global_val_acc, 'r', label='Global Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim([0, 1])
    if save_path:
        plt.show()
        plt.savefig(save_path)
    else:
        plt.show()


def plot_client_accuracies(epochs, client_accuracies, title='Client Local Training Accuracies', save_path=None):
    plt.figure()
    for client_id, accuracies in client_accuracies.items():
        plt.plot(epochs, accuracies, label=f'Client {client_id} Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim([0, 1])
    if save_path:
        plt.show()
        plt.savefig(save_path)
    else:
        plt.show()


def plot_test_results(iid_acc, bias_neutral_acc, bias_conflicting_acc, title='Test Results', save_path=None):
    categories = ['IID', 'Bias Neutral', 'Bias Conflicting']
    accuracies = [iid_acc, bias_neutral_acc, bias_conflicting_acc]

    plt.figure()
    plt.bar(categories, accuracies, color=['blue', 'green', 'red'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    if save_path:
        plt.show()
        plt.savefig(save_path)
    else:
        plt.show()


def read_log_file(log_file_path):
    with open(log_file_path) as f:
        lines = f.readlines()

    epochs = []
    train_loss = []
    val_loss = []
    val_acc = []
    client_accuracies = {}
    iid_acc = None
    bias_neutral_acc = None
    bias_conflicting_acc = None

    for line in lines:
        if "Client" in line and "Local Training Accuracy" in line:
            parts = line.split(", ")
            client_info = parts[0].split("Client ")[1]
            client_id = int(client_info.split(",")[0])
            local_accuracy = float(parts[1].split(": ")[1])
            epoch = len(epochs) + 1  # Assuming this line appears once per epoch per client

            if client_id not in client_accuracies:
                client_accuracies[client_id] = []
            client_accuracies[client_id].append(local_accuracy)

        if "Epoch:" in line and "Train Loss:" in line:
            try:
                parts = line.split(", ")
                epoch_part = parts[0].split(': ')[1]
                train_loss_part = parts[1].split(': ')[1]
                val_loss_part = parts[2].split(': ')[1]
                val_acc_part = parts[3].split(': ')[1]

                epoch = int(epoch_part)
                train_loss_value = float(train_loss_part)
                val_loss_value = float(val_loss_part)
                val_acc_value = float(val_acc_part)

                epochs.append(epoch)
                train_loss.append(train_loss_value)
                val_loss.append(val_loss_value)
                val_acc.append(val_acc_value)
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line.strip()}. Error: {e}")

        if "IID Test Accuracy" in line:
            iid_acc = float(line.split("IID Test Accuracy: ")[1])
        if "Bias-Neutral Test Accuracy" in line:
            bias_neutral_acc = float(line.split("Bias-Neutral Test Accuracy: ")[1])
        if "Bias-Conflicting Test Accuracy" in line:
            bias_conflicting_acc = float(line.split("Bias-Conflicting Test Accuracy: ")[1])

    return epochs, train_loss, val_loss, val_acc, client_accuracies, iid_acc, bias_neutral_acc, bias_conflicting_acc



if __name__ == "__main__":
    log_file_path = "results/log_federated_Exp-1_Epochs-10_Clients-4_L2-False_strat-FedAvg_localIters-1.log"  # Update the path as needed
    epochs, train_loss, val_loss, val_acc, client_accuracies, iid_acc, bias_neutral_acc, bias_conflicting_acc = read_log_file(
        log_file_path)

    exp = "fl1"

    print(f"Epochs: {epochs}")  # Debug print
    print(f"Train Loss: {train_loss}")  # Debug print
    print(f"Val Loss: {val_loss}")  # Debug print
    print(f"Val Accuracy: {val_acc}")  # Debug print
    print(f"Client Accuracies: {client_accuracies}")  # Debug print
    print(f"IID Accuracy: {iid_acc}")  # Debug print
    print(f"Bias Neutral Accuracy: {bias_neutral_acc}")  # Debug print
    print(f"Bias Conflicting Accuracy: {bias_conflicting_acc}")  # Debug print

    # Plot global model accuracy
    plot_global_accuracy(epochs, val_acc, title=f'Global Model Accuracy - {exp}', save_path=f'results/global_accuracy-{exp}.png')

    # Plot client local accuracies
    plot_client_accuracies(epochs, client_accuracies, title=f'Client Local Training Accuracies - {exp}',
                           save_path=f'results/client_accuracies-{exp}.png')

    # Plot test results
    plot_test_results(iid_acc, bias_neutral_acc, bias_conflicting_acc, title=f'Test Results - {exp}',
                      save_path=f'results/test_results-{exp}.png')
