
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generar 300 observaciones
X, y = make_moons(
    n_samples=300,
    noise=0.20,
    random_state=42
)

# Dividir 70% entrenamiento y 30% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

valores_k = [1, 5, 15]
modelos = {}

for k in valores_k:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)

    modelos[k] = modelo

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)

    print(f"k = {k}")
    print(f"  Accuracy entrenamiento: {acc_train:.4f}")
    print(f"  Accuracy prueba: {acc_test:.4f}")
    
# Graficar los datos originales
plt.figure(figsize=(7, 5))

plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap="coolwarm",
    edgecolors="k", alpha=0.8
)

plt.title("Datos generados con make_moons")
plt.xlabel("Caracteristica X1")
plt.ylabel("Caracteristica X2")
plt.grid(alpha=0.3)
plt.show()

# Crear una malla para las regiones de decision
x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

puntos_malla = np.c_[xx.ravel(), yy.ravel()]

# Dibujar las regiones para cada valor de k
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, k in zip(axes, valores_k):
    modelo = modelos[k]

    Z = modelo.predict(puntos_malla)
    Z = Z.reshape(xx.shape)

    ax.contourf(
        xx, yy, Z,
        cmap="coolwarm", alpha=0.35
    )

    ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_train, cmap="coolwarm",
        edgecolors="k", s=25
    )

    ax.set_title(f"Regiones de decision k={k}")
    ax.set_xlabel("Caracteristica X1")
    ax.set_ylabel("Caracteristica X2")

plt.tight_layout()
plt.show()
