import matplotlib.pyplot as plt
import numpy as np
import os

# consistent color palette
colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
]

arquitecture = 'Federated'
#arquitecture = 'Centralized'
#optimizer = 'Adam'
optimizer = 'SGD'


def plot_global_accuracy_across_experiments(experiments, title='Global Model Accuracy Across Experiments',
                                            save_path=None):
    plt.figure(figsize=(10, 6))

    for i, (exp_name, data) in enumerate(experiments.items()):
        epochs = data['epochs']
        global_val_acc = data['val_acc']
        std_dev = data['std_dev']

        plt.errorbar(epochs, global_val_acc, yerr=std_dev, label=f'{exp_name}',
                     capsize=5, fmt='-o', color=colors[i], linewidth=2)
        print(f'{exp_name} - bias_neutral accuracy: {global_val_acc}. standard deviation: {std_dev}')

    plt.title(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left', fontsize=6, bbox_to_anchor=(0.98, 0.7))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_bias_aligned_accuracy_across_experiments(experiments, title='Bias Aligned Accuracy Across Experiments',
                                                  save_path=None):
    plt.figure(figsize=(10, 6))

    for i, (exp_name, data) in enumerate(experiments.items()):
        epochs = data['epochs']
        bias_aligned_acc = data['bias_aligned_acc']
        std_dev = data['std_dev_bias_aligned']

        plt.errorbar(epochs, bias_aligned_acc, yerr=std_dev, label=f'{exp_name}',
                     capsize=5, fmt='-o', color=colors[i], linewidth=2)
        print(f'{exp_name} - bias_aligned accuracy: {bias_aligned_acc}. standard deviation: {std_dev}')

    plt.title(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left', fontsize=6, bbox_to_anchor=(0.98, 0.7))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_bias_conflicting_accuracy_across_experiments(experiments, title='Bias Conflicting Accuracy Across Experiments',
                                                      save_path=None):
    plt.figure(figsize=(10, 6))

    for i, (exp_name, data) in enumerate(experiments.items()):
        epochs = data['epochs']
        bias_conflicting_acc = data['bias_conflicting_acc']
        std_dev = data['std_dev_bias_conflicting']

        plt.errorbar(epochs, bias_conflicting_acc, yerr=std_dev, label=f'{exp_name}',
                     capsize=5, fmt='-o', color=colors[i], linewidth=2)
        print(f'{exp_name} - cMNIST B accuracy: {bias_conflicting_acc}. standard deviation: {std_dev}')

    plt.title(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left', fontsize=6, bbox_to_anchor=(0.98, 0.7))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_client_accuracies(epochs, client_accuracies, std_devs=None, title='Client Local Training Accuracies',
                           save_path=None):
    plt.figure(figsize=(10, 6))

    for i, (client_id, mean_accuracies) in enumerate(client_accuracies.items()):
        if std_devs and client_id in std_devs:
            std_dev = std_devs[client_id]
            plt.errorbar(epochs, mean_accuracies, yerr=std_dev, label=f'Client {client_id} Mean Accuracy',
                         capsize=5, fmt='-o', color=colors[i % len(colors)], linewidth=2)
        else:
            plt.plot(epochs, mean_accuracies, label=f'Client {client_id} Mean Accuracy', color=colors[i % len(colors)],
                     linewidth=2)
        print(f'Client {client_id} accuracy: {mean_accuracies} - standard deviation: {std_dev}')

    plt.title(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_test_results(iid_acc, bias_neutral_acc, bias_conflicting_acc, cmnist_c_acc, std_devs=None, title='Test Results',
                      save_path=None):
    categories = ['IID', 'Bias Neutral', 'Bias Conflicting', 'C MNIST_C']
    accuracies = [iid_acc, bias_neutral_acc, bias_conflicting_acc, cmnist_c_acc]

    if std_devs:
        std_iid = std_devs.get('iid_acc', 0)
        std_bias_neutral = std_devs.get('bias_neutral_acc', 0)
        std_bias_conflicting = std_devs.get('bias_conflicting_acc', 0)
        std_cmnist_c = std_devs.get('cmnist_c_acc', 0)
        std_values = [std_iid, std_bias_neutral, std_bias_conflicting, std_cmnist_c]
    else:
        std_values = [0, 0, 0, 0]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracies, yerr=std_values, capsize=10, color=colors[:4], edgecolor='black')
    plt.title(title, fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    for bar, accuracy, std in zip(bars, accuracies, std_values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{accuracy:.2f} ± {std:.2f}',
                 ha='center', va='bottom', fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
    cmnist_c_acc = None
    bias_aligned_acc = []  # To store bias-aligned accuracies per epoch
    bias_conflicting_acc_per_epoch = []  # To store bias-conflicting accuracies per epoch

    for line in lines:
        if "Client" in line and "Local Training Accuracy" in line:
            parts = line.split(", ")
            client_info = parts[0].split("Client ")[1]
            client_id = int(client_info.split(",")[0])
            local_accuracy = float(parts[1].split(": ")[1])

            if client_id not in client_accuracies:
                client_accuracies[client_id] = []
            client_accuracies[client_id].append(local_accuracy)

        if "Epoch:" in line and "Train Loss:" in line:
            try:
                parts = line.split(", ")
                epoch_part = parts[0].split(': ')[1].split('/')[0].strip()  # Get current epoch number
                train_loss_part = parts[1].split(': ')[1]
                val_loss_part = parts[2].split(': ')[1]
                val_acc_part = parts[3].split(': ')[1]
                bias_aligned_part = parts[4].split(': ')[1]
                bias_conflicting_part = parts[5].split(': ')[1]

                epoch = int(epoch_part)  # Convert only the current epoch number to int
                train_loss_value = float(train_loss_part)
                val_loss_value = float(val_loss_part)
                val_acc_value = float(val_acc_part)
                bias_aligned_value = float(bias_aligned_part)
                bias_conflicting_value = float(bias_conflicting_part)

                epochs.append(epoch)
                train_loss.append(train_loss_value)
                val_loss.append(val_loss_value)
                val_acc.append(val_acc_value)
                bias_aligned_acc.append(bias_aligned_value)
                bias_conflicting_acc_per_epoch.append(bias_conflicting_value)
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line.strip()}. Error: {e}")

        if "IID Test Accuracy" in line:
            iid_acc = float(line.split("IID Test Accuracy: ")[1])
        if "Bias-Neutral Test Accuracy" in line:
            bias_neutral_acc = float(line.split("Bias-Neutral Test Accuracy: ")[1])
        if "Bias-Conflicting Test Accuracy" in line:
            bias_conflicting_acc = float(line.split("Bias-Conflicting Test Accuracy: ")[1])
        if "C MNIST_C Test Accuracy" in line:
            cmnist_c_acc = float(line.split("C MNIST_C Test Accuracy: ")[1])

    return epochs, train_loss, val_loss, val_acc, client_accuracies, iid_acc, bias_neutral_acc, bias_conflicting_acc, cmnist_c_acc, bias_aligned_acc, bias_conflicting_acc_per_epoch


def aggregate_experiment_data(log_files):
    aggregated_data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'client_accuracies': {},
        'iid_acc': [],
        'bias_neutral_acc': [],
        'bias_conflicting_acc': [],
        'cmnist_c_acc': [],
        'bias_aligned_acc': [],
        'bias_conflicting_acc_per_epoch': []
    }

    for log_file in log_files:
        epochs, train_loss, val_loss, val_acc, client_accuracies, iid_acc, bias_neutral_acc, bias_conflicting_acc, cmnist_c_acc, bias_aligned_acc, bias_conflicting_acc_per_epoch = read_log_file(
            log_file)

        if epochs:  # Ensure epochs are not empty
            aggregated_data['epochs'].append(epochs)
        if train_loss:
            aggregated_data['train_loss'].append(train_loss)
        if val_loss:
            aggregated_data['val_loss'].append(val_loss)
        if val_acc:
            aggregated_data['val_acc'].append(val_acc)
        if bias_aligned_acc:
            aggregated_data['bias_aligned_acc'].append(bias_aligned_acc)
        if bias_conflicting_acc_per_epoch:
            aggregated_data['bias_conflicting_acc_per_epoch'].append(bias_conflicting_acc_per_epoch)

        for client_id, accuracies in client_accuracies.items():
            if client_id not in aggregated_data['client_accuracies']:
                aggregated_data['client_accuracies'][client_id] = []
            aggregated_data['client_accuracies'][client_id].append(accuracies)

        if iid_acc is not None:
            aggregated_data['iid_acc'].append(iid_acc)
        if bias_neutral_acc is not None:
            aggregated_data['bias_neutral_acc'].append(bias_neutral_acc)
        if bias_conflicting_acc is not None:
            aggregated_data['bias_conflicting_acc'].append(bias_conflicting_acc)
        if cmnist_c_acc is not None:
            aggregated_data['cmnist_c_acc'].append(cmnist_c_acc)

    return aggregated_data


def compute_means_and_stds(aggregated_data):
    means = {
        'epochs': np.mean(aggregated_data['epochs'], axis=0) if aggregated_data['epochs'] else [],
        'train_loss': np.mean(aggregated_data['train_loss'], axis=0) if aggregated_data['train_loss'] else [],
        'val_loss': np.mean(aggregated_data['val_loss'], axis=0) if aggregated_data['val_loss'] else [],
        'val_acc': np.mean(aggregated_data['val_acc'], axis=0) if aggregated_data['val_acc'] else [],
        'client_accuracies': {},
        'iid_acc': np.mean([x for x in aggregated_data['iid_acc'] if x is not None]) if aggregated_data[
            'iid_acc'] else None,
        'bias_neutral_acc': np.mean([x for x in aggregated_data['bias_neutral_acc'] if x is not None]) if
        aggregated_data['bias_neutral_acc'] else None,
        'bias_conflicting_acc': np.mean([x for x in aggregated_data['bias_conflicting_acc'] if x is not None]) if
        aggregated_data['bias_conflicting_acc'] else None,
        'cmnist_c_acc': np.mean([x for x in aggregated_data['cmnist_c_acc'] if x is not None]) if
        aggregated_data['cmnist_c_acc'] else None,
        'bias_aligned_acc': np.mean(aggregated_data['bias_aligned_acc'], axis=0) if aggregated_data[
            'bias_aligned_acc'] else [],
        'bias_conflicting_acc_per_epoch': np.mean(aggregated_data['bias_conflicting_acc_per_epoch'], axis=0) if
        aggregated_data['bias_conflicting_acc_per_epoch'] else []
    }

    std_devs = {
        'epochs': np.std(aggregated_data['epochs'], axis=0) if aggregated_data['epochs'] else [],
        'train_loss': np.std(aggregated_data['train_loss'], axis=0) if aggregated_data['train_loss'] else [],
        'val_loss': np.std(aggregated_data['val_loss'], axis=0) if aggregated_data['val_loss'] else [],
        'val_acc': np.std(aggregated_data['val_acc'], axis=0) if aggregated_data['val_acc'] else [],
        'client_accuracies': {},
        'iid_acc': np.std([x for x in aggregated_data['iid_acc'] if x is not None]) if aggregated_data[
            'iid_acc'] else None,
        'bias_neutral_acc': np.std([x for x in aggregated_data['bias_neutral_acc'] if x is not None]) if
        aggregated_data['bias_neutral_acc'] else None,
        'bias_conflicting_acc': np.std([x for x in aggregated_data['bias_conflicting_acc'] if x is not None]) if
        aggregated_data['bias_conflicting_acc'] else None,
        'cmnist_c_acc': np.std([x for x in aggregated_data['cmnist_c_acc'] if x is not None]) if
        aggregated_data['cmnist_c_acc'] else None,
        'bias_aligned_acc': np.std(aggregated_data['bias_aligned_acc'], axis=0) if aggregated_data[
            'bias_aligned_acc'] else [],
        'bias_conflicting_acc_per_epoch': np.std(aggregated_data['bias_conflicting_acc_per_epoch'], axis=0) if
        aggregated_data['bias_conflicting_acc_per_epoch'] else []
    }

    for client_id, client_accuracies in aggregated_data['client_accuracies'].items():
        if client_accuracies:  # Ensure client_accuracies is not empty
            means['client_accuracies'][client_id] = np.mean(client_accuracies, axis=0)
            std_devs['client_accuracies'][client_id] = np.std(client_accuracies, axis=0)

    return means, std_devs


def read_all_log_files(log_folder, experiment_numbers, num_trials):
    all_log_files = []

    for exp_num in experiment_numbers:
        trial_files = []
        for trial in range(1, num_trials + 1):
            clients = 4
            if exp_num == 9:
                clients = 3
            if arquitecture == 'Federated':
                file_name = f"log_{arquitecture.lower()}_Exp-{exp_num}_Epochs-10_Clients-{clients}_L2-False_strat-FedAvg_{trial}_{optimizer}.log"
            else:
                file_name = f"log_{arquitecture.lower()}_exp{exp_num}_10_False_{trial}_{optimizer}.log"
            log_file_path = os.path.join(log_folder, file_name)
            if os.path.exists(log_file_path):
                trial_files.append(log_file_path)
            else:
                print(f"File not found: {log_file_path}")
        all_log_files.append(trial_files)

    return all_log_files


def plot_cmnist_c_accuracy_across_experiments(experiments, title='C MNIST_C Test Accuracy Across Experiments',
                                              save_path=None):
    plt.figure(figsize=(10, 6))

    experiment_names = list(experiments.keys())
    cmnist_c_accuracies = [data['cmnist_c_acc'] for data in experiments.values()]
    std_devs = [data['std_dev_cmnist_c'] for data in experiments.values()]
    print(f'cMNIST C accuracies: {cmnist_c_accuracies} - Standard deviations: {std_devs}')

    bars = plt.bar(range(1, len(experiment_names) + 1), cmnist_c_accuracies, yerr=std_devs, capsize=5,
                   color=colors[:len(experiment_names)], edgecolor='black', label=experiment_names)

    plt.title(title, fontsize=20)
    plt.xlabel('Experiments', fontsize=16)
    plt.ylabel('C MNIST_C Test Accuracy', fontsize=16)
    plt.ylim([0, 1])

    plt.xticks(ticks=range(1, len(experiment_names) + 1), labels=range(1, len(experiment_names) + 1), rotation=0,
               fontsize=12)

    plt.yticks(fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.legend(bars, experiment_names, loc='upper right', fontsize=8)

    for bar, accuracy, std in zip(bars, cmnist_c_accuracies, std_devs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{accuracy:.2f} ± {std:.2f}', ha='center',
                 va='bottom', fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    experiment_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_trials = 3
    log_folder = "results"  # Update the path as needed

    all_log_files = read_all_log_files(log_folder, experiment_numbers, num_trials)

    experiments = {}

    for i, log_files in enumerate(all_log_files, start=1):
        if not log_files:
            continue  # Skip if no log files found for this experiment

        aggregated_data = aggregate_experiment_data(log_files)
        means, std_devs = compute_means_and_stds(aggregated_data)

        exp_name = f"Experiment {i}"
        experiments[exp_name] = {
            'epochs': means['epochs'],
            'val_acc': means['val_acc'],
            'std_dev': std_devs['val_acc'],
            'bias_aligned_acc': means['bias_aligned_acc'],
            'std_dev_bias_aligned': std_devs['bias_aligned_acc'],
            'bias_conflicting_acc': means['bias_conflicting_acc_per_epoch'],
            'std_dev_bias_conflicting': std_devs['bias_conflicting_acc_per_epoch'],
            'cmnist_c_acc': means['cmnist_c_acc'],  # Added this for plotting C MNIST_C Accuracy
            'std_dev_cmnist_c': std_devs['cmnist_c_acc']  # Standard deviation for C MNIST_C
        }

        print(f'Experiment name: {exp_name}')
        # Plot client local accuracies
        plot_client_accuracies(means['epochs'], means['client_accuracies'], std_devs['client_accuracies'],
                               title=f'Client Local Training Accuracies - {exp_name} - {optimizer}',
                               save_path=f'final_graphs/client_accuracies-{exp_name}_{optimizer}.png')

        # Plot test results including C MNIST_C accuracy
        plot_test_results(means['iid_acc'], means['bias_neutral_acc'], means['bias_conflicting_acc'],
                          means['cmnist_c_acc'], std_devs,
                          title=f'Test Results - {exp_name}', save_path=f'final_graphs/test_results-federated_{exp_name}_{optimizer}.png')

    # Plot global model accuracy across all experiments
    plot_global_accuracy_across_experiments(experiments,
                                            title=f'MNIST Accuracy Across Experiments - {arquitecture} - {optimizer}',
                                            save_path=f'final_graphs/global_accuracy_across_experiments_federated_{optimizer}.png')

    # Plot bias-aligned accuracy across all experiments
    plot_bias_aligned_accuracy_across_experiments(experiments,
                                                  title=f'cMNIST-A Accuracy Across Experiments - {arquitecture} - {optimizer}',
                                                  save_path=f'final_graphs/bias_aligned_accuracy_across_experiments_federated_{optimizer}.png')

    # Plot bias-conflicting accuracy across all experiments
    plot_bias_conflicting_accuracy_across_experiments(experiments,
                                                      title=f'cMNIST-B Accuracy Across Experiments - {arquitecture} - {optimizer}',
                                                      save_path=f'final_graphs/bias_conflicting_accuracy_across_experiments_federated_{optimizer}.png')

    # Plot C MNIST_C accuracy across all experiments
    plot_cmnist_c_accuracy_across_experiments(experiments,
                                              title=f'C MNIST_C Test Accuracy Across Experiments - {arquitecture} - {optimizer}',
                                              save_path=f'final_graphs/cmnist_c_accuracy_across_experiments_federated_{optimizer}.png')