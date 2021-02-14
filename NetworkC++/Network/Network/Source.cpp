#include <vector>
#include <cstdlib>

#include "NNetwork.h"

void ReadData(std::vector<std::vector<double>>& matrix, std::vector<double>& gender);

void main()
{
	NNetwork network(20);

    std::vector<std::vector<double>> data;
    //= {
    //    {-2, -1, 18}, // �����
    //    {25, 6, 28 }, // ���
    //    {17, 4, 36},  // �����
    //    {-15, -6, 20} // �����
    //};

    std::vector<double> all_y_trues;

    ReadData(data, all_y_trues);
    //= {
    //    1, // �����
    //    0, // ���
    //    0, // �����
    //    1  // ����� 
    //};

    network.train(data, all_y_trues);

    std::vector<double> emily = { -7, -3, 19 };
    std::vector<double> frank = { 20, 2, 20 };
    std::cout << "Emily: " << network.feedforward(emily) << std::endl;
    std::cout << "Frank: " << network.feedforward(frank) << std::endl;
}