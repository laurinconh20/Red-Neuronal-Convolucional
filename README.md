# Clasificación del estado de frutas con CNN

Proyecto de Deep Learning para clasificar imágenes de manzanas, bananos y naranjas en seis categorías: frescas y podridas.

## Clases

- `freshapples`: manzana fresca
- `freshbanana`: banano fresco
- `freshoranges`: naranja fresca
- `rottenapples`: manzana podrida
- `rottenbanana`: banano podrido
- `rottenoranges`: naranja podrida

## Estructura

```text
proyecto-frutas-cnn/
├── app/
│   └── streamlit_app.py
├── data/
├── models/
│   └── modelo_frutas_final.keras
├── notebooks/
│   └── clasificacion_frutas_cnn.ipynb
├── results/
├── src/
│   ├── predict.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenamiento

Abrir `notebooks/clasificacion_frutas_cnn.ipynb` en Google Colab, ejecutar todas las celdas y descargar `modelo_frutas_final.keras`. Colocar el modelo dentro de `models/`.

## Ejecutar Streamlit

Desde la carpeta raíz:

```bash
streamlit run app/streamlit_app.py
```

## Predicción desde consola

```bash
python src/predict.py ruta/a/imagen.jpg
```

## Resultados

El modelo obtuvo una precisión aproximada del **98 %** en el conjunto de prueba, con un **F1-score promedio de 0.98**. Las curvas de entrenamiento, la matriz de confusión, el reporte de clasificación y las predicciones visuales se encuentran en 
otebooks/clasificacion_frutas_cnn.ipynb.
