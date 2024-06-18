import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from PySpice.Unit import *
from PySpice.Probe.Plot import plot

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

import torch
from torchvision import datasets, transforms

# data loader
from torch.utils.data import DataLoader
from schmittTrigger import SchmittTrigger
from snn_circuit import SNNModel
# from app.Neural_Network import loadMNIST
from sys import argv

data_path='/tmp/data/mnist'
def loadMNIST():
    transform = transforms.Compose([
                # transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    return mnist_train, mnist_test

if __name__ == '__main__':
    if len(argv) < 2:
        filename = 'Net_mnist_model.json'
        # filename = 'Net_model.json'
    else:
        filename = argv[1]

    train_loader, test_loader = loadMNIST()

    train_loader = DataLoader(train_loader, batch_size=100, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_loader, batch_size=100, shuffle=True, drop_last=True)
    # get a sample of 100 images and labels from the test set
    test_batch = iter(test_loader)
    images, labels = next(test_batch)
    # print(labels)
    images = images.numpy()
    images = images[:100]
    labels = labels.numpy()
    labels = labels[:100]

    correct = 0
    for i in range(100):
        circuit = Circuit('SNN Circuit')
        input_ = images[i].reshape(28*28)
        # convert label to one-hot encoding
        label = labels[i]
        target = np.zeros(10)
        target[label] = 1
        # print(input_)
        # print(target)
        circuit.include("uopamp_v1.1.lib")
        circuit.subcircuit(SNNModel('snn', filename))
        for i in range(len(input_)):
            circuit.V(f'input_{i}', f'input_{i}', circuit.gnd, input_[i]@u_V)
        
        circuit.X('snn', 'snn', *[f'input_{i}' for i in range(len(input_))], *[f'output_{i}' for i in range(10)])
        # add load resistances to each output
        for i in range(10):
            circuit.R(f'load_{i}', f'output_{i}', circuit.gnd, 1@u_kΩ)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=1@u_ms, end_time=10@u_ms)
        # save the max value found at each output
        outputs = []
        for i in range(10):
            outputs.append(np.max(analysis.nodes[f'output_{i}']))
        # print(np.argmax(outputs), label)
        if np.argmax(outputs) == label:
            correct += 1
    print("The circuit obtained an accuracy of", correct, "% on the test set.")

