import matplotlib.pyplot as plt
import numpy as np
import os

# Consistent color palette
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

optimizer = 'Adam'
#optimizer = 'SGD'


def read_centralized_log_file(log_file_path):
    with open(log_file_path) as f:
        lines = f.readlines()

    epochs = []
    train_loss = []
    val_loss = []
    val_acc = []
    bias_aligned_acc = []  # To store bias-aligned accuracies per epoch
    bias_conflicting_acc = []  # To store bias-conflicting accuracies per epoch
    cmnist_c_acc = None  # To store cMNIST C Test accuracy

    for line in lines:
        if "Epoch:" in line and "Train Loss:" in line:
            try:
                parts = line.split(", ")
                epoch_part = parts[0].split(': ')[1].split('/')[0]
                train_loss_part = parts[1].split(': ')[1]
                val_loss_part = parts[2].split(': ')[1]
                val_acc_part = parts[3].split(': ')[1]
                bias_aligned_part = parts[4].split(': ')[1]
                bias_conflicting_part = parts[5].split(': ')[1]

                epoch = int(epoch_part)
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
                bias_conflicting_acc.append(bias_conflicting_value)
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line.strip()}. Error: {e}")

        if "cMNIST C Test accuracy" in line:
            cmnist_c_acc = float(line.split("cMNIST C Test accuracy ")[1])

    return epochs, val_acc, bias_aligned_acc, bias_conflicting_acc, cmnist_c_acc


def aggregate_experiment_data(log_files):
    max_epochs = 0
    aggregated_data = {
        'epochs': [],
        'val_acc': [],
        'bias_aligned_acc': [],
        'bias_conflicting_acc': [],
        'cmnist_c_acc': []  # Adding this line
    }

    for log_file in log_files:
        epochs, val_acc, bias_aligned_acc, bias_conflicting_acc, cmnist_c_acc = read_centralized_log_file(log_file)
        max_epochs = max(max_epochs, len(epochs))
        aggregated_data['epochs'].append(epochs)
        aggregated_data['val_acc'].append(val_acc)
        aggregated_data['bias_aligned_acc'].append(bias_aligned_acc)
        aggregated_data['bias_conflicting_acc'].append(bias_conflicting_acc)
        if cmnist_c_acc is not None:
            aggregated_data['cmnist_c_acc'].append(cmnist_c_acc)

    # Padding the sequences to have the same length
    for key in ['epochs', 'val_acc', 'bias_aligned_acc', 'bias_conflicting_acc']:
        for i in range(len(aggregated_data[key])):
            if len(aggregated_data[key][i]) < max_epochs:
                diff = max_epochs - len(aggregated_data[key][i])
                aggregated_data[key][i] += [np.nan] * diff  # Fill with NaN

    return aggregated_data


def compute_means_and_stds(aggregated_data):
    means = {
        'epochs': np.nanmean(aggregated_data['epochs'], axis=0),
        'val_acc': np.nanmean(aggregated_data['val_acc'], axis=0),
        'bias_aligned_acc': np.nanmean(aggregated_data['bias_aligned_acc'], axis=0),
        'bias_conflicting_acc': np.nanmean(aggregated_data['bias_conflicting_acc'], axis=0),
        'cmnist_c_acc': np.nanmean(aggregated_data['cmnist_c_acc']) if aggregated_data['cmnist_c_acc'] else None
    }

    std_devs = {
        'val_acc': np.nanstd(aggregated_data['val_acc'], axis=0),
        'bias_aligned_acc': np.nanstd(aggregated_data['bias_aligned_acc'], axis=0),
        'bias_conflicting_acc': np.nanstd(aggregated_data['bias_conflicting_acc'], axis=0),
        'cmnist_c_acc': np.nanstd(aggregated_data['cmnist_c_acc']) if aggregated_data['cmnist_c_acc'] else None
    }

    return means, std_devs


def read_all_log_files(log_folder, experiment_numbers, num_trials):
    all_log_files = []

    for exp_num in experiment_numbers:
        trial_files = []
        for trial in range(1, num_trials + 1):
            file_name = f"log_centralized_exp{exp_num}_10_False_{trial}_{optimizer}.log"
            log_file_path = os.path.join(log_folder, file_name)
            if os.path.exists(log_file_path):
                trial_files.append(log_file_path)
            else:
                print(f"File not found: {log_file_path}")
        all_log_files.append(trial_files)

    return all_log_files


def plot_experiment_data(experiments, title='Accuracy Across Experiments', y_label='Accuracy', save_path=None):
    plt.figure(figsize=(12, 8))

    for i, (exp_name, data) in enumerate(experiments.items()):
        epochs = data['epochs']
        mean_acc = data['val_acc']  # Mean accuracy across trials
        std_dev = data['std_dev_val_acc']  # Standard deviation across trials
        print(f'{exp_name} - accuracy: {mean_acc} - stddev: {std_dev}')

        # Plotting mean accuracy with error bars representing the standard deviation
        plt.errorbar(epochs, mean_acc, yerr=std_dev, label=exp_name, capsize=5, fmt='-o',
                     color=colors[i % len(colors)], linewidth=2)

    plt.title(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_cmnist_c_accuracy_across_experiments(experiments, title='cMNIST C Test Accuracy Across Experiments', save_path=None):
    plt.figure(figsize=(10, 6))

    experiment_names = list(experiments.keys())
    cmnist_c_accuracies = [data['cmnist_c_acc'] for data in experiments.values()]
    std_devs = [data['std_dev_cmnist_c'] for data in experiments.values()]
    print(f'cMNIST C accuracies: {cmnist_c_accuracies} - Standard deviation: {std_devs}\n')

    # Use range(1, len(experiment_names) + 1) for x-axis values
    bars = plt.bar(range(1, len(experiment_names) + 1), cmnist_c_accuracies, yerr=std_devs, capsize=5,
                   color=colors[:len(experiment_names)], edgecolor='black')

    plt.title(title, fontsize=20)
    plt.xlabel('Experiments', fontsize=16)
    plt.ylabel('C MNIST_C Test Accuracy', fontsize=16)
    plt.ylim([0, 1])

    # Set custom x-axis labels to show 1-9
    plt.xticks(ticks=range(1, len(experiment_names) + 1), labels=range(1, len(experiment_names) + 1), rotation=0, fontsize=12)

    plt.yticks(fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    for bar, accuracy, std in zip(bars, cmnist_c_accuracies, std_devs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{accuracy:.2f} ± {std:.2f}', ha='center', va='bottom', fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    experiment_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_trials = 3
    log_folder = "results"

    all_log_files = read_all_log_files(log_folder, experiment_numbers, num_trials)
    experiments = {}

    for i, log_files in enumerate(all_log_files, start=1):
        if not log_files:
            continue

        aggregated_data = aggregate_experiment_data(log_files)
        means, std_devs = compute_means_and_stds(aggregated_data)

        exp_name = f"Exp {i}"
        experiments[exp_name] = {
            'epochs': means['epochs'],
            'val_acc': means['val_acc'],
            'std_dev_val_acc': std_devs['val_acc'],
            'bias_aligned_acc': means['bias_aligned_acc'],
            'std_dev_bias_aligned': std_devs['bias_aligned_acc'],
            'bias_conflicting_acc': means['bias_conflicting_acc'],
            'std_dev_bias_conflicting': std_devs['bias_conflicting_acc'],
            'cmnist_c_acc': means['cmnist_c_acc'],  # Adding cMNIST C Test accuracy
            'std_dev_cmnist_c': std_devs['cmnist_c_acc']  # Adding standard deviation for cMNIST C Test accuracy
        }
    # Plot validation accuracy across all experiments
    print('MNIST accuracies')
    plot_experiment_data(
        experiments,
        title=f'Centralized Accuracies - Centralized - {optimizer}',
        y_label='Validation Accuracy',
        save_path=f'final_graphs/centralized_val_accuracy_across_experiments_{optimizer}.png'
    )

    print('cMNIST A accuracies')
    # Plot bias-aligned accuracy across all experiments
    plot_experiment_data(
        {exp_name: {'epochs': data['epochs'], 'val_acc': data['bias_aligned_acc'], 'std_dev_val_acc': data['std_dev_bias_aligned']}
         for exp_name, data in experiments.items()},
        title=f'cMNIST-A Accuracies - Centralized - {optimizer}',
        y_label='cMNIST-A Accuracies',
        save_path=f'final_graphs/centralized_bias_aligned_accuracy_across_experiments_{optimizer}.png'
    )

    print('cMNIST B accuracies')
    # Plot bias-conflicting accuracy across all experiments
    plot_experiment_data(
        {exp_name: {'epochs': data['epochs'], 'val_acc': data['bias_conflicting_acc'], 'std_dev_val_acc': data['std_dev_bias_conflicting']}
         for exp_name, data in experiments.items()},
        title=f'cMNIST-B Accuracies - Centralized - {optimizer}',
        y_label='cMNIST-B Accuracies',
        save_path=f'final_graphs/centralized_bias_conflicting_accuracy_across_experiments_{optimizer}.png'
    )

    print('cMNIST C accuracies')
    # Plot cMNIST C Test accuracy across all experiments
    plot_cmnist_c_accuracy_across_experiments(
        experiments,
        title=f'cMNIST C Final accuracy across experiments - Centralized - {optimizer}',
        save_path=f'final_graphs/cmnist_c_accuracy_across_experiments_centralized_{optimizer}.png'
    )
