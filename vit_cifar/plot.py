import matplotlib.pyplot as plt
import re

def parse_log_file(filename):
    steps = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    
    with open(filename, 'r') as file:
        idx = -1
        for line in file:
            match = re.search(r'step (\d+), loss: ([\d\.]+), test_loss: ([\d\.]+), acc: ([\d\.]+), test_acc: ([\d\.]+)', line)
            if match:
                curr_steps = int(match.group(1))
                if curr_steps == 50:
                    idx += 1
                steps.append(curr_steps + 750 * idx)
                train_loss.append(float(match.group(2)))
                test_loss.append(float(match.group(3)))
                train_acc.append(float(match.group(4)))
                test_acc.append(float(match.group(5)))
    
    return steps, train_loss, test_loss, train_acc, test_acc

# File paths
files = {
    "Adam": "cifar_logs/adam_cifar10_lr_0.0005.txt",
    "delta-GClip": "cifar_logs/dgclip_cifar10_lr_0.2_gamma_1.0.txt",
    # "SGD": "cifar_logs/sgd_cifar10_lr_0.1.txt"
}

# lr_list = [0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.5]
lr_list = [0.2]
# lr_list = [0.1]
# gamma_list = [1.0, 1.5, 2.0]
gamma_list = [1.0]
plot_option = 'Train'

# files = {f'lr_{lr}_gamma_{gamma}': f'cifar_logs/dgclip_cifar10_lr_{lr}_gamma_{gamma}.txt' for lr in lr_list for gamma in gamma_list}

# Dictionary to store parsed data
data = {}
for key, file in files.items():
    data[key] = parse_log_file(file)

# Plot training and test loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# breakpoint()
for key in data:
    # if plot_option.lower() == 'train':
        plt.plot(data[key][0], data[key][1], label=f'{key} Train Loss')
    # elif plot_option.lower() == 'test':
        plt.plot(data[key][0], data[key][2], linestyle='dashed', label=f'{key} Test Loss')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
# plt.title(f'{plot_option} Loss (lr={lr_list[0]})')
# plt.title(f'{plot_option} Loss')
plt.legend()

# Plot training and test accuracy
plt.subplot(1, 2, 2)
for key in data:
    # if plot_option.lower() == 'train':
        plt.plot(data[key][0], data[key][3], label=f'{key} Train Acc')
    # elif plot_option.lower() == 'test':
        plt.plot(data[key][0], data[key][4], linestyle='dashed', label=f'{key} Test Acc')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
# plt.title(f'{plot_option} Accuracy (lr={lr_list[0]})')
# plt.title(f'{plot_option} Accuracy')
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig(f'cifar_logs/plot_hp_lr={lr_list[0]}_gamma={gamma_list[0]}.pdf')
