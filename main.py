import numpy as np

from SiteParser import site_parser
from NeuralNetwork import OurNeuralNetwork
from FilePlitter import get_training_data, get_tests_data, show_results_after_tests


def main():
    #  get data from site
    outputFileName = site_parser()

    #  prepare data to network format
    data, gender = get_training_data(outputFileName)
    data = np.array(data)
    gender = np.array(gender)

    #  create network and train it
    network = OurNeuralNetwork(10)
    network.train(data, gender)

    # tests
    data, gender = get_tests_data()
    data = np.array(data)
    testsResultFile = 'tests_result.txt'
    with open(testsResultFile, 'w') as file:
        for i in range(len(data)):
            file.write('{networkAnswer:^0.3f} : {height} : {mass} : {age} : {realValue}\n'.format(
                networkAnswer=network.feedforward(data[i]),
                height=data[i][0],
                mass=data[i][1],
                age=data[i][2],
                realValue=gender[i]))

    #  show results
    show_results_after_tests(testsResultFile)


if __name__ == '__main__':
    main()
