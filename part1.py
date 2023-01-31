# Importing standard Qiskit libraries and configuring account
import qiskit as qk
from qiskit import QuantumCircuit, Aer, IBMQ, QuantumRegister
from qiskit import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import plot_histogram
from math import pi
import numpy as np
from qiskit.circuit.library.standard_gates import RYGate
import itertools

def encoder(image):
    '''take image and return circuit encoding'''
    old = image
    #convert 28*28 image to 14*14 for added efficiency
    new = np.zeros((14,14))
    for i in range(14):
        for j in range(14):
            new[i,j] = (old[2*i,2*j]+old[2*i+1,2*j]+old[2*i,2*j+1]+old[2*i+1,2*j+1])/4
    
    #define theta array from 0 to pi/2 to encode pixel intensities
    theta = (pi/2)*new/(np.max(new))
    theta = np.pad(theta, ((1, 1),(1,1)), 'constant')
    theta = theta.flatten()
 
    #create circuit
    qr = QuantumRegister(9)
    qc = QuantumCircuit(qr)

    qc.h(np.arange(0,9))
    qc.barrier()

    #Pixel 1

    cry_gate = RYGate(theta[0]*2).control(8)
    qc.append(cry_gate,qr)

    qc.barrier()

    def pixelStorage():
        pixelCount = 1
        while pixelCount < 256:
            n = 8
            p = 1
            while n != 0:
                n=n-1
                if pixelCount%p == 0:
                    qc.x(n)
                    p = p+p

            cry_gate = RYGate(theta[pixelCount]*2).control(8)
            qc.append(cry_gate,qr)

            qc.barrier() 

            pixelCount += 1

    pixelStorage()

    qc.measure_all()
    
    return qc

def simulator(circuit):
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, aer_sim)
    qobj = assemble(t_qc, shots=4096)
    result = aer_sim.run(qobj).result()
    counts = result.get_counts(qc)
    return counts

def decode(histogram):
    bw = np.zeros(512) 
    i = 0
    j = 0
    while j < 512:
        for i in itertools.product([0,1], repeat=9):
            #white
            try:
                t = counts[str(i[0])+str(i[1])+str(i[2])+str(i[3])+str(i[4])+str(i[5])+str(i[6])+str(i[7])+str(i[8])]
                bw[j] = t
            except:
                bw[j] = 0
            #t = counts[str(i[0])+str(i[1])+str(i[2])+str(i[3])+str(i[4])+str(i[5])+str(i[6])+str(i[7])+str(i[8])]
            #if t:
            #    bw[j] = t
            #else:
            #    bw[j] = 0
        
            j+=1
    w = decode(counts)[:256]
    b = decode(counts)[256:]
    intensity = w-b
    intensity = 255*(intensity/np.max(np.abs(intensity))+1)/2
    reg_img = intensity.reshape((16,16))
    return reg_img

def run_part1(image):
    circuit = encoder(image)
    histogram = simulate(circuit)
    image = decoder(histogram)
    return circuit, image