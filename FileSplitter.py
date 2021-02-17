dataFile = 'dataFile.txt'
testFile = 'testFile.txt'


def get_training_data(fileName: str):
    # get data from file
    with open(fileName, 'r') as file:
        allData = file.readlines()

    # data training and tests in ratio 75/25
    dataCount = int((len(allData) - 1) * 3 / 4)
    data = []
    gender = []
    testsData = []
    testsGender = []
    # split data to training
    for i in range(1, dataCount):
        splittedParams = allData[i].split(';')
        data.append([float(splittedParams[0]), float(splittedParams[1])])
        gender.append(int(splittedParams[2]))

    # split data to tests
    for i in range(dataCount, len(allData)):
        splittedParams = allData[i].split(';')
        testsData.append([float(splittedParams[0]), float(splittedParams[1])])
        testsGender.append(int(splittedParams[2]))

    # write training data into file
    with open(dataFile, 'w') as file:
        file.write('{0:^8} {1:^6} {2:^6}\n'.format('height', 'mass', 'gender'))
        for i in range(len(data)):
            file.write('{0:^8.2f} {1:^6.2f} {2:^6}\n'.format(data[i][0], data[i][1],
                                                                    'Female' if gender[i] else 'Male'))

    # write tests data into file
    with open(testFile, 'w') as file:
        file.write('{0:^8} {1:^6} {2:^6}\n'.format('height', 'mass', 'gender'))
        for i in range(len(testsData)):
            file.write('{0:^8.2f}:{1:^6.2f}:{2:^6}\n'.format(testsData[i][0], testsData[i][1],
                                                                    'Female' if testsGender[i] else 'Male'))

    return data, gender


def get_tests_data():
    data = []
    gender = []

    # read tests data
    with open(testFile, 'r') as file:
        testsData = file.readlines()
        for i in range(1, len(testsData)):
            values = testsData[i].split(':')
            data.append([float(values[0]), float(values[1])])
            gender.append(values[2] == 'Female\n')

    return data, gender


def show_results_after_tests(resultsFileName: str):
    results = None
    mistakes = []
    yes = 0
    no = 0

    # My vision of error
    sumValues = 0.0

    # read results data
    with open(resultsFileName, 'r') as file:
        results = file.readlines()
    for i in range(len(results)):
        values = results[i].split(':')

        # My vision of error
        # sumValues += float(values[0])
        # Or
        sumValues += round(float(values[0]))

        networkAnswer = None
        if float(values[0]) > 0.8:
            networkAnswer = 'Female'
        elif float(values[0]) < 0.2:
            networkAnswer = 'Male'
        else:
            networkAnswer = 'closer to ' + ('Female' if float(values[0]) >= 0.5 else 'Male')

        realValue = 'Female' if values[3][1:] == 'True\n' else 'Male'
        if networkAnswer == realValue:
            yes += 1
        else:
            no += 1
        mistakes.append('Height: {height} Mass:{mass} Gender:{gender} Neural answer:{neuralAnswer}'.format
            (
            height=values[1],
            mass=values[2],
            gender=realValue,
            neuralAnswer=networkAnswer
        ))

    # show results
    if no == 0:
        print('All {count} was successes defined'.format(count=yes))
    else:
        # print('Fail is {percent}%\n'.format(percent=no / (len(results) / 100)))
        print('Fail is {percent}%\n'.format(percent=100 - sumValues / (len(results) / 100)))
    for i in range(len(mistakes)):
        print(mistakes[i])


def read_trainings_data():
    weights = []
    bias = []
    with open("trainingsWeights", 'r') as file:
        for value in file.readlines():
            weights.append(float(value))
    with open("trainingsBias", 'r') as file:
        for value in file.readlines():
            bias.append(float(value))
    return weights, bias