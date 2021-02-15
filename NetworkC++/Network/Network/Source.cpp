#include <vector>
#include <cstdlib>
#include <fstream>

#include "NNetwork.h"

void ReadData(std::vector<std::vector<double>>& matrix, std::vector<double>& gender);

int main()
{
	NNetwork network(3);

    std::vector<std::vector<double>> data;
    std::vector<double> all_y_trues;

    ReadData(data, all_y_trues);


    data = network.TranslationAndNormalization(data);

    network.train(data, all_y_trues, 0.00001, 4010);

    std::ofstream file("trainingsWeights");

    std::vector<double> weights; // = return from network

    if (file.is_open())
    {
        for (const auto& value : weights)
            file << value << std::endl;
    }

    return 0;
}