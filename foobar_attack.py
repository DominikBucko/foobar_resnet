from sage.all import *

import torch
import numpy as np
from resnet_remote import ResNet18
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load a trained network

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18()
net = net.to(device)
net.eval()

model = "./trained_models/resnet18_1_rows_1_channels_0.5_probability.pth"
checkpoint = torch.load(model, map_location=torch.device(device))
state_dict = OrderedDict((k.removeprefix('module.'), v) for k, v in checkpoint['net'].items())
net.load_state_dict(state_dict)

# Load dataset

transform_t = transforms.Compose([
    transforms.ToTensor(),
])

# Normalization function for the samples from the dataset
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

trainset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform_t)


def toPair(i,size):
    return (i//size,i%size)

def toIndex(p,size):
    i = p[0]
    j = p[1]
    if (i < 0) or (j<0) or (i>(size-1)) or (j>(size-1)):
        return -1
    return (i*size+j)


def findIndexes(i,size):
    (a,b) = toPair(i,size)

    LT = toIndex((a-1,b-1),size)
    CT = toIndex((a-1,b),size)
    RT = toIndex((a-1,b+1),size)
    L= toIndex((a,b-1),size)
    C= toIndex((a,b),size)
    R= toIndex((a,b+1),size)
    LB = toIndex((a+1,b-1),size)
    CB=toIndex((a+1,b),size)
    RB=toIndex((a+1,b+1),size)

    Indexes = [LT,CT,RT,L,C,R,LB,CB,RB]

    return Indexes


def index3dto1d(x, y, z):
    xMax = 32
    yMax = 32
    return (z * xMax * yMax) + (y * xMax) + x


def solveConstraints(attack_img, net):
    W = net.conv1.weight[1].detach().numpy()
    w = list(W.flatten())
    bn_mean = float(net.bn1.running_mean[1])
    bn_var = float(net.bn1.running_var[1])
    bn_weight = float(net.bn1.weight[1])
    bn_bias = float(net.bn1.bias[1])

    # means and std for every channel, calculated from the entire dataset
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)

    p = MixedIntegerLinearProgram()
    x = p.new_variable(real=True, nonnegative=False)

    cols = 32*32*3


    #Inputs between 0 and 1
    if (attack_img is not None):
        # Constraints for 'pattern' image
        for pos in range(cols):
            p.add_constraint(x[pos] <= min(attack_img[pos]+0.7, 1))
            p.add_constraint(x[pos] >= max(attack_img[pos]-0.7, 0))

    else:
        #Constraints for 'free' image
        for pos in range(cols):
            p.add_constraint(x[pos] <= 1)
            p.add_constraint(x[pos] >= 0)

    #Constraints on Matrix multiplication
    #assuming size=32x32
    for i in range(0,32,2):
        for j in range(0,32,2):
            indexes = findIndexes(toIndex((i, j), 32), 32)
            p.add_constraint(
                (((sum(
                    (x[indexes[pos%9] + 32*32 * int(pos/9)] - means[int(pos/9)]) / stds[int(pos/9)] * w[pos] if indexes[pos%9] != -1 else 0 for pos in range(27)
                ) - bn_mean) / bn_var) * bn_weight) + bn_bias <= 0
            )

    try:
        p.solve()
        s = vector([p.get_values(x[pos]) for pos in range(cols)])
        s = np.array(s)
    except:
        print('No solution found')
        s = None
    return s


base_img = np.array(trainset[0][0]).transpose(2,0,1).flatten()


res = solveConstraints(base_img, net)

res = solveConstraints(None, net)

if not res:
    exit(0)

test_input = torch.from_numpy(res.reshape(1, 3, 32, 32)).type(torch.FloatTensor)

# It is necessary to normalize the input to be in range
test_input = normalize(test_input)

with torch.inference_mode():
    output = net(test_input)
    if list(output[0]).index(max(output[0])) == checkpoint["fault_config"]["target_class"]:
        print("Attack successful. Saving output image...")
        plt.imsave("fooling_image.png", res.reshape(3,32,32).transpose(2,0,1))
    else:
        print("Attack failed.")