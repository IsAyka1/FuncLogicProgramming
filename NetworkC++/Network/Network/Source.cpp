#include <vector>
#include <cstdlib>

#include "NNetwork.h"

void ReadData(std::vector<std::vector<double>>& matrix, std::vector<double>& gender);

void main()
{
	NNetwork network(3);

    std::vector<std::vector<double>> data;
    //data = {
    //    {-2, -1, 18}, // Алиса
    //    {25, 6, 28 }, // Боб
    //    {17, 4, 36},  // Чарли
    //    {-15, -6, 20} // Диана
    //};

    std::vector<double> all_y_trues;

    ReadData(data, all_y_trues);
    //all_y_trues = {
    //    1, // Алиса
    //    0, // Боб
    //    0, // Чарли
    //    1  // Диана 
    //};

    data = network.TranslationAndNormalization(data);

    network.train(data, all_y_trues, 0.001, 10000);

    //std::vector<double> Weights = network.m_Weights, Bias = network.m_Bias;

    std::vector<double> emily = { -6, -30, 63 };  //{ 151.76, 47.83, 63.0 };  //{ -6, -30, 63 };
    std::vector<double> frank = { -11, -54, 63 };   //{ 139.70, 36.49, 63.0 };  //{ -11, -54, 63 };
    std::cout << "Emily: " << network.feedforward(emily) << std::endl;
    std::cout << "Frank: " << network.feedforward(frank) << std::endl;
}