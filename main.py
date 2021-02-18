import numpy as np
import os

from SiteParser import site_parser
from NeuralNetwork import OurNeuralNetwork
from FileSplitter import *


def main():
    #  get data from site
    outputFileName = site_parser()

    #  prepare data to network format
    data, gender = get_training_data(outputFileName)
    data = np.array(data)
    gender = np.array(gender)

    hiddenNeurons = '4'
    #  create network and train it
    network = OurNeuralNetwork(int(hiddenNeurons))

    # With Python
    # network.train(data, gender)

    # With C ++
    # os.system('Network.exe ' + hiddenNeurons)
    network.W, network.B = read_trainings_data()  # here initialize network

    # tests
    data, gender = get_tests_data()
    data = np.array(data)
    testsResultFile = 'tests_result.txt'
    with open(testsResultFile, 'w') as file:
        for i in range(len(data)):
            file.write('{networkAnswer:^0.3f} : {height} : {mass} : {realValue}\n'.format(
                networkAnswer=network.feedforward(data[i]),
                height=data[i][0],
                mass=data[i][1],
                realValue=gender[i]))

    #  show results
    show_results_after_tests(testsResultFile)


if __name__ == '__main__':
    main()
