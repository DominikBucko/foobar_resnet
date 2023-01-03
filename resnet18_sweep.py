from resnet import *
from utils import progress_bar

import torch.nn as nn
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os



lr = 0.1
epochs = 40
batch_size = 128

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
#%%
# Define network to train
net_model = ResNet18
#%%
# Attack configuration
number_of_rows_faulted = [32, 28, 24, 20, 17, 14, 11, 10, 8, 7, 6, 5, 4, 3, 2, 1]
fault_probability = [0.5]
number_of_channels_faulted = [1]
target_class = 1


def get_attack_config():
    for rows in number_of_rows_faulted:
        for channels in number_of_channels_faulted:
            for prob in fault_probability:
                yield {
                    "rows": rows,
                    "channels": channels,
                    "probability": prob,
                    "target_class": target_class,
                }
#%%



device = 'cuda' if torch.cuda.is_available() else 'cpu'

for config in get_attack_config():
    print(config)

    best_acc = 0

    # Model
    print('==> Building model..')
    net = net_model()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    # Training
    def train(e):
        print('\nEpoch: %d' % e)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs, targets.tolist(), config, False)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    def test(e):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print(
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': e,
                'fault_config': config
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f"./checkpoint/resnet18_{config['rows']}_rows_{config['channels']}_channels_{config['probability']}_probability.pth")
            best_acc = acc


    def test_faults():
        net.eval()
        total = 0
        faults_successful = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs_faulted = net(inputs, targets, config, test=True)
                output_classes = [list(i).index(max(i)) for i in outputs_faulted]
                faults_successful += output_classes.count(1)
                total += targets.size(0)

            print(f"Faults successful = {faults_successful}/{total}")


    for epoch in range(0, epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
    test_faults()

