from matplotlib import pyplot as plt

with open("results/log_federated_mnist_10_3_2") as f:
    fed = f.readlines()

fed_val = [float(f.split(" ")[7].strip(",")) for f in fed if "Train" in f]
fed_train = [float(f.split(" ")[4].strip(",")) for f in fed if "Train" in f]

epochs = range(1, 11)
plt.figure()
plt.plot(epochs, fed_train, 'g', label='Training loss')
plt.plot(epochs, fed_val, 'b', label='validation loss')
plt.title('Federated loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks()
plt.ylim([0,1])
