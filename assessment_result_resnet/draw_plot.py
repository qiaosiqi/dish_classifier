import matplotlib.pyplot as plt

# 数据
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss = [4.4365, 4.0559, 3.7058, 3.3755, 3.0165, 2.6916, 2.0163, 1.5132, 1.1329, 0.4654]
val_acc = [0.0728, 0.0927, 0.1308, 0.1991, 0.2355, 0.2956, 0.3392, 0.4365, 0.5932, 0.8361]

# Plot Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, marker='o', label='Loss')
plt.axvline(6.5, linestyle='--', color='gray', alpha=0.6)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy', color='green')
plt.axvline(6.5, linestyle='--', color='gray', alpha=0.6)
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
