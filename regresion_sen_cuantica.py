import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Dispositivo cuántico
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def quantum_model(x, weights):
    qml.RX(x * weights[0], wires=0)
    return qml.expval(qml.PauliZ(0))

def predict(x, weights):
    return weights[1] * quantum_model(x, weights) + weights[2]

def cost(weights, X, Y):
    predictions = [predict(x, weights) for x in X]
    return np.mean((np.array(predictions) - Y) ** 2)

# Datos sinusoidales
X = np.linspace(0, 2 * np.pi, 20)
Y = np.sin(X)

np.random.seed(0)
weights = np.random.randn(3)  # Solo 3 pesos para mayor rapidez

opt = qml.AdamOptimizer(stepsize=0.05)  # Adam es más rápido y estable
for _ in range(500):  # Menos iteraciones para velocidad
    weights = opt.step(lambda w: cost(w, X, Y), weights)

predictions = [predict(x, weights) for x in X]

plt.scatter(X, Y, label="Datos sinusoidales")
plt.plot(X, predictions, label="Ajuste Cuántico Sinusoidal Optimizado", linewidth=2)
plt.xlabel("X")
plt.ylabel("sin(X)")
plt.title("Regresión Cuántica Sinusoidal Rápida con PennyLane")
plt.legend()
plt.grid(True)
plt.show()
