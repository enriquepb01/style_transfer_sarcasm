import pickle
import matplotlib.pyplot as plt

# Unpickle file
filename = 'history_last.pickle'
results_file = open(filename, 'rb')
results = pickle.load(results_file)
results_file.close()

epoch_length = len(results['train_loss'])

# Training and validation loss
plt.figure()
plt.plot(range(epoch_length), results['train_loss'])
plt.plot(range(epoch_length), results['validation_loss'])
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.ylim([0, 1.4])
plt.legend(['Training', 'Validation'])
plt.show()

# Training and validation accuracy
plt.figure()
plt.plot(range(epoch_length), results['train_accuracy'])
plt.plot(range(epoch_length), results['validation_accuracy'])
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(['Training', 'Validation'])
plt.show()