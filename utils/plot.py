import matplotlib.pyplot as plt
import numpy as np

train_acc = [0.53, 0.64, 0.70, 0.72, 0.75]
val_acc = [0.45, 0.52, 0.60]
test_acc = [0.51, 0.60, 0.68]
#x = np.linspace(0, 2, 100)

plt.plot(train_acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.plot(test_acc, label='test_acc')

plt.xlabel('Iteration')
plt.ylabel('Acc')

plt.title("Training Acc")

plt.legend()

plt.show()