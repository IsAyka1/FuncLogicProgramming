#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "NNetwork.h"

void ReadData(std::vector<std::vector<double>>& matrix, std::vector<double>& gender);

int main(int argc, char* argv[])
{
    std::size_t count;
    if (argc > 1)
    {
        std::istringstream iss(argv[1]);
        iss >> count;
    }

	NNetwork network(count);

    std::vector<std::vector<double>> data;
    std::vector<double> all_y_trues;

    ReadData(data, all_y_trues);


    data = network.TranslationAndNormalization(data);

    network.train(data, all_y_trues, 0.00001, 4010);

    std::ofstream weightsFile("trainingsWeights"), biasFile("trainingsBias");

    std::vector<double> weights = network.m_Weights, bias = network.m_Bias; // = return from network

    if (weightsFile.is_open())
    {
        for (const auto& value : weights)
            weightsFile << value << std::endl;
    }
    if (biasFile.is_open())
    {
        for (const auto& value : bias)
            biasFile << value << std::endl;
    }

    return 0;
}