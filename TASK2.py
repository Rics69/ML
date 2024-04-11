import numpy as np
import matplotlib.pyplot as plt

# Cлучайные данные
np.random.seed(0)
X = np.random.rand(100, 2)

# Инициализация центроидов
k = 3
centroids = X[:k, :]

# Функция для вычисления расстояния между точками и центроидами
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Начальное отображение точек и центроидов
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x')
plt.title('Шаг 0')
plt.show()

# Алгоритм k-means
for step in range(1, 6):  # Пять шагов
    clusters = [[] for _ in range(k)]

    # Назначение точек к ближайшим центроидам
    for point in X:
        distances = [distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(point)

    # Пересчет центроидов
    for i in range(k):
        centroids[i] = np.mean(clusters[i], axis=0)

    # Отображение точек и центроидов на текущем шаге
    colors = ['blue', 'green', 'orange']
    plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in np.argmin([distance(X, centroid) for centroid in centroids], axis=0)])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x')
    plt.title(f'Шаг {step}')
    plt.show()