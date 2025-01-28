# Proyecto: Optimización de RandomForest para Detección de Billetes Falsos

## Descripción
Este repositorio contiene un modelo de clasificación basado en **RandomForestClassifier** optimizado mediante **GridSearchCV** y **RandomizedSearchCV**. El objetivo es clasificar billetes genuinos y falsos a partir de características numéricas extraídas del dataset.

## Contenido del Repositorio
- `modelo_random_forest.pkl` → Modelo optimizado con **GridSearchCV**.
- `modelo_random_forest_randomized.pkl` → Modelo optimizado con **RandomizedSearchCV**.
- `notebook.ipynb` → Cuaderno Jupyter con el código de entrenamiento y optimización.
- `LICENSE` → Licencia MIT para el uso del código.
- `README.md` → Explicación del proyecto y cómo usar los modelos.

## Datos Utilizados
El dataset utilizado contiene información sobre billetes reales y falsos, con atributos como dimensiones, márgenes y características de impresión. La columna objetivo (`is_genuine`) indica si un billete es auténtico (`True`) o falso (`False`).

## Optimización del Modelo
Se aplicaron dos enfoques para la optimización de hiperparámetros:
1. **GridSearchCV** → Exploración exhaustiva de combinaciones de hiperparámetros.
2. **RandomizedSearchCV** → Búsqueda aleatoria con 25 iteraciones.

Los mejores hiperparámetros encontrados para el modelo optimizado fueron:
- `n_estimators = 100`
- `max_depth = 20`
- `min_samples_split = 10`
- `min_samples_leaf = 2`
- `max_features = 'log2'`

## Resultados
Ambos modelos lograron una **precisión del 98.67%** en el conjunto de prueba, asegurando una clasificación confiable de billetes falsos y genuinos.

## Uso del Modelo Guardado
Para cargar y usar el modelo en un entorno de Python:
```python
import joblib
modelo = joblib.load("modelo_random_forest.pkl")
resultado = modelo.predict(nuevos_datos)
