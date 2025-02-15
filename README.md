
# Que hace el codigo

El clasificacion.py implementa clasificadores cuánticos utilizando Qiskit Machine Learning. Primero, genera un conjunto de datos aleatorios de entrenamiento con dos características y etiquetas asignadas según la suma de esas características. Luego, crea y entrena dos clasificadores cuánticos: uno basado en un EstimatorQNN, que calcula valores esperados de estados cuánticos, y otro basado en un SamplerQNN, que muestrea mediciones repetidas de un circuito cuántico. Ambos clasificadores utilizan el optimizador COBYLA, con la posibilidad de ajustar parámetros como el número de iteraciones y repeticiones de puertas cuánticas. Durante el entrenamiento, se visualiza en tiempo real la evolución de la función objetivo, y al final, se muestran los resultados de clasificación, incluyendo los errores. El código ofrece flexibilidad para optimizar el rendimiento mediante ajustes en el ansatz cuántico y el optimizador.

El red_neuronal_artificial_ml.py utiliza la biblioteca Qiskit para implementar un clasificador cuántico variacional (VQC). Primero, se establece una semilla para la reproducibilidad y se cargan datos de un conjunto ad hoc, dividiéndolos en características y etiquetas para entrenamiento y prueba. Luego, se define un mapa de características cuántico (ZZFeatureMap) y un ansatz (TwoLocal), que representan el estado inicial y las operaciones cuánticas a aplicar. El clasificador VQC se crea combinando estos elementos junto con el optimizador COBYLA para ajustar los parámetros. Finalmente, el modelo se entrena con los datos de entrenamiento y se evalúa con los datos de prueba, mostrando la precisión obtenida.



# Crear y activar el entorno virtual

/virtualenv env
/virtualenv -p "C:\Archivos de programas\python310\python.exe" env # para python con otra version
/env/Scripts/activate.bat

# requerimientos

pip install qiskit-machine-learning

pip install matplotlib

pip install pylatexenc

pip install pyqt5

pip install pennylane


# Ejecución
py .\clasificacion.py
