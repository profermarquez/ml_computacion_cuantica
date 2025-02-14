import matplotlib.pyplot as plt
import numpy as np
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA    # otros optimizadores: SPSA, ADAM
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler


# Fijar la semilla para reproducibilidad
algorithm_globals.random_seed = 42

# Generar datos de entrenamiento
num_inputs = 2
num_samples = 20
X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
y01 = 1 * (np.sum(X, axis=1) >= 0)  # en {0, 1}
y = 2 * y01 - 1  # en {-1, +1}

# Inicializar lista para valores de la función objetivo
valores_funcion_objetivo = []

# Función de callback para seguimiento del optimizador
def graficar_seguimiento(pesos, evaluacion_funcion_objetivo):
    valores_funcion_objetivo.append(evaluacion_funcion_objetivo)
    plt.clf()
    plt.title("Valor de la función objetivo vs. iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Valor de la función objetivo")
    plt.plot(range(len(valores_funcion_objetivo)), valores_funcion_objetivo)
    plt.pause(0.1)

# Configuración de Matplotlib
plt.ion()
plt.rcParams["figure.figsize"] = (12, 6)

# Visualizar datos de entrenamiento
plt.figure()
for x, y_target in zip(X, y):
    if y_target == 1:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "go")
plt.plot([-1, 1], [1, -1], "--", color="black")
plt.title("Datos de entrenamiento")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

# Crear circuito cuántico utilizando QNNCircuit
qc = QNNCircuit(num_qubits=num_inputs)

pass_manager = PassManager()
# Crear EstimatorQNN
estimator = StatevectorEstimator()
estimator_qnn = EstimatorQNN(circuit=qc, estimator=estimator)

# Crear clasificador
estimator_classifier = NeuralNetworkClassifier(
    estimator_qnn, optimizer=COBYLA(maxiter=40), callback=graficar_seguimiento
)

# Entrenar clasificador
estimator_classifier.fit(X, y)

# Evaluar clasificador
score = estimator_classifier.score(X, y)
print(f"Puntuación del clasificador: {score}")

# Predecir etiquetas
y_predict = estimator_classifier.predict(X)

# Visualizar resultados
plt.figure()
for x, y_target, y_p in zip(X, y, y_predict):
    if y_target == 1:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "go")
    if y_target != y_p:
        plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
plt.plot([-1, 1], [1, -1], "--", color="black")
plt.title("Resultados de la clasificación")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

# Imprimir pesos del clasificador
print("Pesos del clasificador:", estimator_classifier.weights)

# Crear nuevo circuito cuántico con un ansatz específico
ansatz = RealAmplitudes(num_inputs, reps=6) # puedo mejorar aumentando las repeticiones de la compuerta
qc = QNNCircuit(ansatz=ansatz)

# Definir función de paridad
def paridad(x):
    return "{:b}".format(x).count("1") % 2

# Crear SamplerQNN
sampler = StatevectorSampler()
sampler_qnn = SamplerQNN(
    circuit=qc,
    interpret=paridad,
    output_shape=2,
    sampler=sampler,
)

# Crear clasificador con SamplerQNN
sampler_classifier = NeuralNetworkClassifier(
    neural_network=sampler_qnn, optimizer=COBYLA(maxiter=37), callback=graficar_seguimiento
)

# Reiniciar lista de valores de la función objetivo
valores_funcion_objetivo = []

# Entrenar clasificador con SamplerQNN
sampler_classifier.fit(X, y01)

# Evaluar clasificador con SamplerQNN
score_sampler = sampler_classifier.score(X, y01)
print(f"Puntuación del clasificador con SamplerQNN: {score_sampler}")

# Predecir etiquetas con SamplerQNN
y_predict_sampler = sampler_classifier.predict(X)

# Visualizar resultados con SamplerQNN
plt.figure()
for x, y_target, y_p in zip(X, y01, y_predict_sampler):
    if y_target == 1:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "go")
    if y_target != y_p:
        plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
plt.plot([-1, 1], [1, -1], "--", color="black")
plt.title("Resultados de la clasificación con SamplerQNN")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show(block=True)
