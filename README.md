# 🧠 Recomendador de Tenis Jordan con CLIP

## Descripción del Proyecto
Este proyecto utiliza el modelo CLIP (Contrastive Language-Image Pretraining) de OpenAI para crear un sistema de recomendación de tenis Jordan. El sistema puede:

- 🔍 **Clasificar** una imagen de tenis en 31 modelos diferentes de Jordan
- 🛍️ **Recomendar** los tenis más parecidos visualmente del catálogo
- 📝 **Describir** las características visuales detectadas en la imagen

## Características Principales

### ✨ Data Augmentation
- Utiliza flip horizontal para mayor robustez
- Combina embeddings de imagen original y volteada
- Mejora la invarianza a orientación

### 🎯 Clasificación Multimodal
- Compara embeddings de imagen vs texto
- 31 clases de modelos Jordan predefinidas
- Sistema de confianza basado en similitud coseno

### 🖼️ Análisis Visual Detallado
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

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/pepemt/tarea_clasificacion.git
cd tarea_clasificacion
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta el código:
```bash
python clasificacion_v2.py
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
José Tahuilan

## Conclusiones

El modelo CLIP permite representar imágenes y texto en un mismo espacio semántico, facilitando la clasificación y recomendación basada en similitud visual. El uso de Data Augmentation mejora significativamente la robustez del sistema ante variaciones de orientación.