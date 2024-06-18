import numpy as np
import json
from scipy.linalg import solve
import matplotlib.pyplot as plt
from PySpice.Unit import *
from PySpice.Probe.Plot import plot
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

from schmittTrigger import SchmittTrigger
from snn_circuit import SNNModel
from sys import argv

if __name__ == '__main__':
    if len(argv) < 2:
        filename = './xor_weights_snn.json'
        # filename = 'Net_model.json'
        # filename = 'Net_xor.json'
    else:
        filename = argv[1]
    correct = 0
    for inputVoltages in [[0, 0], [1, 0], [0, 1], [1, 1]]:
        circuit = Circuit('Xor model')
        circuit.include("uopamp_v1.1.lib")
        snn = SNNModel('snn', filename)
        # print(snn.layers)
        circuit.subcircuit(snn)
        for i in range(len(inputVoltages)):
            circuit.V(f'input_{i}', f'input_{i}', circuit.gnd, inputVoltages[i]@u_V)
        # connect the last layer to the output
        circuit.X('snn', 'snn', 'input_0', 'input_1', 'output_0')
        # add resistance load to final output
        circuit.R(f'load', 'output_0', circuit.gnd, 1@u_kΩ)
        
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=1@u_ms, end_time=10@u_ms)

        maxVoltage = np.max(np.array(analysis['output_0']))
        if maxVoltage > 0.5:
            if np.sum(inputVoltages) == 1:
                correct += 1
        else:
            if np.sum(inputVoltages) == 0 or np.sum(inputVoltages) == 2:
                correct += 1
    print("The circuit obtained an accuracy of", correct*100/4, "% on the test set.")
    