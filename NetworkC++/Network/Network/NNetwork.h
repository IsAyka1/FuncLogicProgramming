#include <vector>
#include <cstdlib>
#include <iostream>
#include <ctime>

#define _USE_MATH_DEFINES
#include <cmath>

class NNetwork
{
	int m_HiddenNeurons;
	std::vector<double> m_Weights = {}, m_Bias = {};

	// Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
	double sigmoid(double in_x)
	{
		return 1 / (1 + exp(-in_x));
	}

	// Производная сигмоиды : f'(x) = f(x) * (1 - f(x))
	double deriv_sigmoid(double in_x)
	{
		double fx = sigmoid(in_x);
		return fx * (1 - fx);
	}

	// Определение потери MSE
	double mse_loss(const std::vector<double>& in_y_true, const std::vector<double>& in_y_pred)
	{
		double error = 0;
		for (int i = 0; i < in_y_true.size(); i++) {
			error += pow((in_y_true[i] - in_y_pred[i]), 2);
		}
		return error / in_y_pred.size();
	}

	// Получить обрезанную версию вектора
	std::vector<double> cutVec(const std::vector<double>& V, int in_begin, int in_end)
	{
		std::vector<double> result;
		for (int i = in_begin; i < in_end; i++)
			result.push_back(V[i]);
		return result;
	}

	// Поэлементное перемножение векторов
	std::vector<double> multVec(const std::vector<double>& in_V1, const std::vector<double>& in_V2)
	{
		std::vector<double> result;
		for (int i = 0; i < in_V1.size(); i++)
			result.push_back(in_V1[i] * in_V2[1]);
		return result;
	}

	// Поэлементное перемножение вектора на число
	std::vector<double> multVec(const std::vector<double>& in_V, double item)
	{
		std::vector<double> result;
		for (int i = 0; i < in_V.size(); i++)
			result.push_back(in_V[i] * item);
		return result;
	}

	// Сумма всех элементов вектора
	double sumVec(const std::vector<double>& in_V)
	{
		double result = 0;
		for (const auto& it : in_V)
			result += it;
		return result;
	}

public:

	NNetwork(int in_HidNeu) : m_HiddenNeurons(in_HidNeu)
	{
		std::srand(std::time(nullptr)); // use current time as seed for random generator
		int random_variable = ((double)std::rand() / RAND_MAX) * 2 - 0.5;

		for (int i = 0; i < 4 * in_HidNeu; i++)
		{
			m_Weights.push_back(((double)std::rand() / RAND_MAX));
			//m_Weights.push_back(0);
			//std::cout << m_Weights[i] << ' ';
		}
		std::cout << std::endl;
		for (int i = 0; i < in_HidNeu + 1; i++)
		{
			m_Bias.push_back(((double)std::rand() / RAND_MAX));
			//m_Bias.push_back(0);
			//std::cout << m_Bias[i] << ' ';
		}
		std::cout << std::endl;
	}

	double feedforward(const std::vector<double>& in_X)
	{
		std::vector<double> hide = {};
		for (int i = 0; i < m_HiddenNeurons; i++)
		{
			hide.push_back(sigmoid(m_Weights[i * 3] * in_X[0] + m_Weights[i * 3 + 1] * in_X[1] + m_Weights[i * 3 + 2] * in_X[2] + m_Bias[i]));
		}

		double O1 = sigmoid(sumVec(multVec(cutVec(m_Weights, 3 * m_HiddenNeurons, m_Weights.size()), hide)) + m_Bias[m_Bias.size() - 1]);

		return O1;
	}

	void train(const std::vector<std::vector<double>>& data, const std::vector<double>& all_y_trues)
	{
		double learn_rate = 0.5;
		int epochs = 100000;

		for (int epoch = 0; epoch < epochs; epoch++)
		{
			for (int k = 0; k < all_y_trues.size(); ++k)
			{
				std::vector<double> sum_H = {};
				std::vector<double> hide = {};
				for (int i = 0; i < m_HiddenNeurons; ++i)
				{
					sum_H.push_back(m_Weights[i * 3] * data[k][0] + m_Weights[i * 3 + 1] * data[k][1] + m_Weights[i * 3 + 2] * data[k][2] + m_Bias[i]);
					hide.push_back(sigmoid(sum_H[i]));
				}

				double sum_O1 = sumVec(multVec(cutVec(m_Weights, 3 * m_HiddenNeurons, m_Weights.size()), hide)) + m_Bias[m_Bias.size() - 1];
				double O1 = sigmoid(sum_O1);
				double y_pred = O1;

				// --- Считаем частные производные.
				// --- Имена: d_L_d_w1 = "частная производная L по w1"
				double d_L_d_ypred = -2 * (all_y_trues[k] - y_pred);

				std::vector<double> d_ypred_d_H = multVec(cutVec(m_Weights, 3 * m_HiddenNeurons, m_Weights.size()), deriv_sigmoid(sum_O1));
				
				std::vector<double> d_W = {};
				std::vector<double> d_B = {};
				for (int i = 0; i < m_HiddenNeurons; ++i)
				{
					d_W.push_back(data[k][0] * deriv_sigmoid(sum_H[i]));
					d_W.push_back(data[k][1] * deriv_sigmoid(sum_H[i]));
					d_W.push_back(data[k][2] * deriv_sigmoid(sum_H[i]));
					d_B.push_back(deriv_sigmoid(sum_H[i]));
				}

				for (int i = 0; i < m_HiddenNeurons; ++i)
				{
					d_W.push_back(hide[i] * deriv_sigmoid(sum_O1));
				}

				d_B.push_back(deriv_sigmoid(sum_O1));

				// --- Обновляем веса и пороги
				for (int i = 0; i < m_HiddenNeurons; ++i)
				{
					m_Weights[i * 3] -= learn_rate * d_L_d_ypred * d_ypred_d_H[i] * d_W[i * 3];
					m_Weights[i * 3 + 1] -= learn_rate * d_L_d_ypred * d_ypred_d_H[i] * d_W[i * 3 + 1];
					m_Weights[i * 3 + 2] -= learn_rate * d_L_d_ypred * d_ypred_d_H[i] * d_W[i * 3 + 2];
					m_Bias[i] -= learn_rate * d_L_d_ypred * d_ypred_d_H[i] * d_B[i];
				}

				for (int i = m_HiddenNeurons * 3; i < m_HiddenNeurons * 4; ++i)
					m_Weights[i] -= learn_rate * d_L_d_ypred * d_W[i];
				m_Bias[m_HiddenNeurons] -= learn_rate * d_L_d_ypred * d_B[m_HiddenNeurons];
			}

			if (epoch % 10 == 0)
			{
				std::vector<double> y_preds = {};

				for (auto& X : data)
				{
					y_preds.push_back(feedforward(X));
				}
				double loss = mse_loss(all_y_trues, y_preds);
				std::cout << "Epoch " << epoch << " loss: " << loss << std::endl;
			}
		}
	}

};