#include <iostream>
#include <fstream>
#include <string>
#include <vector>

void ReadData(std::vector<std::vector<double>>& matrix, std::vector<double>& gender)
{
	std::string fileName = "dataFile.txt";
	//std::string fileName = "NewData.txt";
	std::ifstream file(fileName);

	matrix.clear();
	gender.clear();

	system("cd");

	if (file.is_open())
	{
		{
			std::string tmp;
			std::getline(file, tmp);
		}

		while (!file.eof())
		{
			std::vector<double> v;
			double value;

			file >> value;
			v.push_back(value);
			file >> value;
			v.push_back(value);
			file >> value;		// Закомментировано при отсутствии возраста в данных
			v.push_back(value);
			matrix.push_back(v);

			std::string sex;
			file >> sex;
			gender.push_back(sex == "Female" ? 1 : 0);
			
		}
		file.close();
	}
	else
	{
		std::cout << "Can not open file " << fileName << std::endl;
	}
}