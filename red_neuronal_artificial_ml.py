from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data

semilla = 1376
algorithm_globals.random_seed = semilla

# Usar el conjunto de datos ad hoc para datos de entrenamiento y prueba
dimension_caracteristica = 2  # dimensión de cada punto de datos
tamano_entrenamiento = 20
tamano_prueba = 10

# características de entrenamiento, etiquetas de entrenamiento, características de prueba, 
# etiquetas de prueba como np.ndarray,
# codificación one-hot para etiquetas
caracteristicas_entrenamiento, etiquetas_entrenamiento, caracteristicas_prueba, etiquetas_prueba = ad_hoc_data(
    training_size=tamano_entrenamiento, test_size=tamano_prueba, n=dimension_caracteristica, gap=0.3
)

mapa_caracteristicas = ZZFeatureMap(feature_dimension=dimension_caracteristica, reps=2, entanglement="linear")
ansatz = TwoLocal(mapa_caracteristicas.num_qubits, ["ry", "rz"], "cz", reps=3)
clasificador_vqc = VQC(
    feature_map=mapa_caracteristicas,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100),
)
clasificador_vqc.fit(caracteristicas_entrenamiento, etiquetas_entrenamiento)

puntuacion = clasificador_vqc.score(caracteristicas_prueba, etiquetas_prueba)


import matplotlib.pyplot as plt
import numpy as np

# Visualizar datos de entrenamiento
plt.figure()
plt.title("Datos de entrenamiento")
plt.scatter(caracteristicas_entrenamiento[:, 0], caracteristicas_entrenamiento[:, 1], c=np.argmax(etiquetas_entrenamiento, axis=1))
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()

# Visualizar datos de prueba
plt.figure()
plt.title("Datos de prueba")
plt.scatter(caracteristicas_prueba[:, 0], caracteristicas_prueba[:, 1], c=np.argmax(etiquetas_prueba, axis=1))
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()

print("Entrenamiento completado.")
print(f"Precisión de prueba luego del entrenamiento, indicado cuan preciso es el modelo: {puntuacion:0.2f}")
