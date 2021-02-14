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
  '''
  Нейронная сеть с:
    - 2 входами
    - скрытым слоем с 2 нейронами (h1, h2)
    - выходной слой с 1 нейроном (o1)

  *** DISCLAIMER ***:
  Следующий код простой и обучающий, но НЕ оптимальный.
  Код реальных нейронных сетей совсем на него не похож. НЕ копируйте его!
  Изучайте и запускайте его, чтобы понять, как работает эта нейронная сеть.
  '''
  def __init__(self, Num):
    # Количество скрытых нейронов
    self.N = Num

    # Веса
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Улучшение
    self.W = np.random.random(4*Num)

    # Пороги
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

    # Улучшение
    self.B = np.random.random(Num+1)

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)

    # Улучшение
    H = np.zeros(self.N)
    for i in range(self.N):
      H[i] = sigmoid(self.W[i*3] * x[0] + self.W[i*3+1] * x[1] + self.W[i*3+2] * x[2] + self.B[i])

    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

    # Улучшение
    O1 = sigmoid((self.W[3*self.N:] * H).sum() + self.B[len(self.B) - 1])

    return O1

  def train(self, data, all_y_trues):
    '''
    - data - массив numpy (n x 2) numpy, n = к-во наблюдений в наборе.
    - all_y_trues - массив numpy с n элементами.
      Элементы all_y_trues соответствуют наблюдениям в data.
    '''
    learn_rate = 0.5
    epochs = 10000 # сколько раз пройти по всему набору данных

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Прямой проход (эти значения нам понадобятся позже)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        # Улучшение
        sum_H = np.zeros(self.N)
        H = np.zeros(self.N)
        for i in range(self.N):
          sum_H[i] = self.W[i*3] * x[0] + self.W[i*3+1] * x[1] + self.W[i*3+2] * x[2] + self.B[i]
          H[i] = sigmoid(sum_H[i])

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        #y_pred = o1

        # Улучшение
        sum_O1 = (self.W[3*self.N:] * H).sum() + self.B[len(self.B)-1]
        O1 = sigmoid(sum_O1)
        y_pred = O1

        # --- Считаем частные производные.
        # --- Имена: d_L_d_w1 = "частная производная L по w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Нейрон o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        # Улучшение
        d_W = np.zeros(self.N * 4)
        d_W[self.N*3:] = H * deriv_sigmoid(sum_O1)
        d_B = np.zeros(self.N + 1)
        d_B[self.N] = deriv_sigmoid(sum_O1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Улучшение
        d_ypred_d_H = self.W[self.N*3:] * deriv_sigmoid(sum_O1)

        # Нейрон h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Нейрон h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # Улучшение
        for i, j in zip(range(0, self.N * 3, 3), range(self.N)):
          d_W[i] = x[0] * deriv_sigmoid(sum_H[j])
          d_W[i+1] = x[1] * deriv_sigmoid(sum_H[j])
          d_W[i+2] = x[2] * deriv_sigmoid(sum_H[j])
          d_B[j] = deriv_sigmoid(sum_H[j])

        # --- Обновляем веса и пороги
        # Нейрон h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Нейрон h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Улучшение
        for i, j in zip(range(0, self.N * 3, 3), range(self.N)):
          self.W[i] -= learn_rate * d_L_d_ypred * d_ypred_d_H[j] * d_W[i]
          self.W[i+1] -= learn_rate * d_L_d_ypred * d_ypred_d_H[j] * d_W[i+1]
          self.W[i+2] -= learn_rate * d_L_d_ypred * d_ypred_d_H[j] * d_W[i+2]
          self.B[j] -= learn_rate * d_L_d_ypred * d_ypred_d_H[j] * d_B[j]

        # Нейрон o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

        # Улучшение
        for i in range(self.N * 3, self.N * 4, 1):
          self.W[i] -= learn_rate * d_L_d_ypred * d_W[i]
        self.B[self.N] -= learn_rate * d_L_d_ypred * d_B[self.N]

      # --- Считаем полные потери в конце каждой эпохи
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.8f" % (epoch, loss))
        #print(self.W)

# Определим набор данных
data = np.array([
  [-2, -1, 18],  # Алиса
  [25, 6, 28],   # Боб
  [17, 4, 36],   # Чарли
  [-15, -6, 20], # Диана
])
all_y_trues = np.array([
  1, # Алиса
  0, # Боб
  0, # Чарли
  1, # Диана
])

# Обучаем нашу нейронную сеть!
network = OurNeuralNetwork(10)
network.train(data, all_y_trues)

# Делаем пару предсказаний
emily = np.array([-7, -3, 19]) # 128 фунтов (52.35 кг), 63 дюйма (160 см)
frank = np.array([20, 2, 20])  # 155 pounds (63.4 кг), 68 inches (173 см)
print("Эмили: %.3f" % network.feedforward(emily)) # 0.951 - Ж
print("Фрэнк: %.3f" % network.feedforward(frank)) # 0.039 - М