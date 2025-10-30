# Recomendador de Tenis Jordan con CLIP

## Descripción del Proyecto
Este proyecto utiliza el modelo CLIP (Contrastive Language-Image Pretraining) de OpenAI para crear un sistema de recomendación de tenis Jordan. El sistema puede:

- **Clasificar** una imagen de tenis en 31 modelos diferentes de Jordan
- **Recomendar** los tenis más parecidos visualmente del catálogo
- **Describir** las características visuales detectadas en la imagen

## Características Principales

### Data Augmentation
- Utiliza flip horizontal para mayor robustez
- Combina embeddings de imagen original y volteada
- Mejora la invarianza a orientación

### Clasificación Multimodal
- Compara embeddings de imagen vs texto
- 31 clases de modelos Jordan predefinidas
- Sistema de confianza basado en similitud coseno

### Análisis Visual Detallado
- Descripción automática de características
- Detección de colores, materiales y estilos
- Top 3 características con porcentajes de confianza

## Estructura del Proyecto

```
tarea1/
├── clasificacion_v2.py    # Código principal
├── requirements.txt       # Dependencias
├── dataset/              # Carpeta con imágenes organizadas por clase
│   ├── Air_Jordan_1/
│   ├── Air_Jordan_3/
│   └── ...
└── README.md            # Este archivo
```


## Uso

```python
# Recomendar tenis similares
recomendar("mi_imagen.jpg", top_k=5)

# Clasificar sin data augmentation
recomendar("mi_imagen.jpg", use_augmentation=False)
```

## Tecnologías Utilizadas

- **PyTorch**: Framework de deep learning
- **CLIP**: Modelo multimodal de OpenAI
- **PIL/Pillow**: Procesamiento de imágenes
- **Matplotlib**: Visualización de resultados
- **NumPy**: Operaciones numéricas

## Autor
- José Tahuilan (A01747922)
- Yose Sotomayor (A01750908)
- Daniel Bernal (A01750047)
- Carlos Zamudio (A01799283)

## Conclusiones

El uso de CLIP demuestra el potencial de los modelos multimodales para unir comprensión visual y semántica textual sin necesidad de entrenamiento supervisado específico para cada clase. Esto permite ampliar fácilmente el sistema a nuevos modelos de tenis simplemente añadiendo descripciones textuales, sin requerir retraining completo.

Gracias a la representación embebida compartida de CLIP, el sistema puede incorporar nuevas imágenes o categorías sin cambios estructurales. Esto lo hace ideal para catálogos en crecimiento o adaaptaciones a nuestra aplicación comerciales donde se agregan productos frecuentemente, manteniendo bajos costos de mantenimiento y entrenamiento. De igual forma, nos permite expandir la tiendaa y no quedarnos en ropa solaamente. Permitiendo tener nuevos usuarios y entrar más de lleno en el mundo del retail.
