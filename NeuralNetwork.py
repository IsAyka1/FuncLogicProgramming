import numpy as np

def sigmoid(x):
  # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true и y_pred - массивы numpy одинаковой длины.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:

  def __init__(self, Num):
    # Количество скрытых нейронов
    self.N = Num
    self.W = np.random.random(3*Num)
    self.B = np.random.random(Num+1)

  def TranslationAndNormalization(self, x):
    result = [x[0] / 2.54 - 66, x[1] * 2.205 - 135]
    return result

  def feedforward(self, value):
    H = np.zeros(self.N)
    x = self.TranslationAndNormalization(value)
    for i in range(self.N):
      H[i] = sigmoid(self.W[i*2] * x[0] + self.W[i*2+1] * x[1] + self.B[i])

    O1 = sigmoid((self.W[2*self.N:] * H).sum() + self.B[len(self.B) - 1])

    return O1

  def train(self, data, all_y_trues):
    learn_rate = 0.00001
    epochs = 4010  # сколько раз пройти по всему набору данных

    for epoch in range(epochs):
      for value, y_true in zip(data, all_y_trues):
        x = self.TranslationAndNormalization(value)

        # --- Прямой проход (эти значения нам понадобятся позже)
        sum_H = np.zeros(self.N)
        H = np.zeros(self.N)
        for i in range(self.N):
          sum_H[i] = self.W[i*2] * x[0] + self.W[i*2+1] * x[1] + self.B[i]
          H[i] = sigmoid(sum_H[i])

        sum_O1 = (self.W[2*self.N:] * H).sum() + self.B[len(self.B)-1]
        O1 = sigmoid(sum_O1)
        y_pred = O1

        # --- Считаем частные производные.
        # --- Имена: d_L_d_ypred = "частная производная L по ypred"
        d_L_d_ypred = -2 * (y_true - y_pred)

        d_W = np.zeros(self.N * 3)
        d_W[self.N*2:] = H * deriv_sigmoid(sum_O1)
        d_B = np.zeros(self.N + 1)
        d_B[self.N] = deriv_sigmoid(sum_O1)

        d_ypred_d_H = self.W[self.N*2:] * deriv_sigmoid(sum_O1)

        for i, j in zip(range(0, self.N * 2, 2), range(self.N)):
          d_W[i] = x[0] * deriv_sigmoid(sum_H[j])
          d_W[i+1] = x[1] * deriv_sigmoid(sum_H[j])
          d_B[j] = deriv_sigmoid(sum_H[j])

        # --- Обновляем веса и пороги
        for i, j in zip(range(0, self.N * 2, 2), range(self.N)):
          self.W[i] -= learn_rate * d_L_d_ypred * d_ypred_d_H[j] * d_W[i]
          self.W[i+1] -= learn_rate * d_L_d_ypred * d_ypred_d_H[j] * d_W[i+1]
          self.B[j] -= learn_rate * d_L_d_ypred * d_ypred_d_H[j] * d_B[j]

        # Нейрон o1
        for i in range(self.N * 2, self.N * 3, 1):
          self.W[i] -= learn_rate * d_L_d_ypred * d_W[i]
        self.B[self.N] -= learn_rate * d_L_d_ypred * d_B[self.N]

      # --- Считаем полные потери в конце каждой эпохи
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.8f" % (epoch, loss))
        #print(self.W)