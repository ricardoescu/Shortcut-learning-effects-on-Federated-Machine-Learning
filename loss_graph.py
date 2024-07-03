import matplotlib.pyplot as plt
import numpy as np

def plot_loss(epochs, train_loss, val_loss, title='Federated Loss', save_path=None):
    plt.figure()
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_accuracy(epochs, val_acc, title='Federated Accuracy', save_path=None):
    plt.figure()
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_time_per_epoch(epochs, epoch_times, title='Time per Epoch', save_path=None):
    plt.figure()
    plt.plot(epochs, epoch_times, 'm', label='Time per epoch')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_cumulative_time(epochs, cumulative_times, title='Cumulative Training Time', save_path=None):
    plt.figure()
    plt.plot(epochs, cumulative_times, 'c', label='Cumulative training time')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Cumulative Time (seconds)')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def read_log_file(log_file_path):
    with open(log_file_path) as f:
        lines = f.readlines()

    part = []
    epochs = []
    train_loss = []
    val_loss = []
    val_acc = []

    for line in lines:
        if "Epoch:" in line and "Train Loss:" in line:
            try:
                parts = line.split(", ")
                epoch_part = parts[0].split(': ')[1]
                train_loss_part = parts[1].split(': ')[1]
                val_loss_part = parts[2].split(': ')[1]
                val_acc_part = parts[3].split(': ')[1]

                print(f"Parsed values - Epoch: {epoch_part}, Train Loss: {train_loss_part}, Val Loss: {val_loss_part}, Val Accuracy: {val_acc_part}")  # Debug print

                epoch = int(epoch_part)
                train_loss_value = float(train_loss_part)
                val_loss_value = float(val_loss_part)
                val_acc_value = float(val_acc_part)

                part.append(parts)
                epochs.append(epoch)
                train_loss.append(train_loss_value)
                val_loss.append(val_loss_value)
                val_acc.append(val_acc_value)
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line.strip()}. Error: {e}")

    return part, epochs, train_loss, val_loss, val_acc

if __name__ == "__main__":
    log_file_path = "results/log_federated_color_mnist_10_2_2"  # Update the path as needed
    part, epochs, train_loss, val_loss, val_acc = read_log_file(log_file_path)

    print(f"Parts: {part}")
    print(f"Epochs: {epochs}")  # Debug print
    print(f"Train Loss: {train_loss}")  # Debug print
    print(f"Val Loss: {val_loss}")  # Debug print
    print(f"Val Accuracy: {val_acc}")  # Debug print

    # Plot loss and accuracy
    plot_loss(epochs, train_loss, val_loss, title='Federated Loss')
    plot_accuracy(epochs, val_acc, title='Federated Accuracy')

    # Assuming you have epoch_times saved in the results file
    results_file_path = f"results/results_color_mnist.npy"  # Update the path as needed
    results = np.load(results_file_path, allow_pickle=True).item()
    epoch_times = results["epoch_times"]
    plot_time_per_epoch(epochs, epoch_times, title='Time per Epoch')
    plot_cumulative_time(epochs, np.cumsum(epoch_times), title='Cumulative Training Time')
