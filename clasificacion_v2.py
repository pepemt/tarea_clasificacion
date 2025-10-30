# ===============================================
# 🧠 Proyecto: Recomendador de Tenis Jordan con CLIP
# Autor: [Tu Nombre]
# Descripción:
#   Este modelo identifica el tipo de tenis Jordan en una imagen
#   y recomienda los más parecidos visualmente dentro de un catálogo.
# ===============================================

import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pprint import pprint

# ======================================================
# 1 Cargar modelo CLIP (modelo preentrenado de OpenAI)
# ======================================================
# Este modelo genera embeddings tanto de texto como de imágenes,
# permitiendo comparar similitud entre ambos dominios.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ======================================================
# 2 Definir ruta del dataset (carpetas por clase)
# ======================================================
DATASET_DIR = "dataset"

# Listas donde se guardarán rutas e identificadores de clase
image_paths = []
labels = []

# Recorrer cada carpeta (una por clase)
for cls in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(cls_path):
        for file in os.listdir(cls_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif')):
                image_paths.append(os.path.join(cls_path, file))
                labels.append(cls)

# ======================================================
# 3 Función para obtener embeddings con Data Augmentation
# ======================================================
# Se usa una versión original y una volteada horizontalmente
# para obtener representaciones más robustas.
def get_embedding(image_path, use_augmentation=True):
    image = Image.open(image_path).convert("RGB")
    
    if use_augmentation:
        # Imagen original
        image_orig = preprocess(image).unsqueeze(0).to(device)
        # Imagen volteada horizontalmente
        image_flipped = preprocess(image.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Embeddings de ambas versiones
            embedding_orig = model.encode_image(image_orig)
            embedding_flipped = model.encode_image(image_flipped)
            # Promedio de ambos embeddings
            embedding = (embedding_orig + embedding_flipped) / 2
    else:
        # Solo imagen original
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
    
    # Normalización para comparaciones de similitud
    return embedding / embedding.norm(dim=-1, keepdim=True)

# ======================================================
# 4 Calcular embeddings del banco de imágenes
# ======================================================
print("Extrayendo embeddings del catálogo con data augmentation...")
embeddings = []
for path in tqdm(image_paths):
    embeddings.append(get_embedding(path, use_augmentation=True))
embeddings = torch.cat(embeddings, dim=0)

# ======================================================
# 5 Definir las 31 clases (modelos de tenis Jordan)
# ======================================================
classes = [
    "Air Jordan 1 Hig h", "Air Jordan 1 Mid", "Air Jordan 1 Low",
    "Air Jordan 2", "Air Jordan 3", "Air Jordan 4", "Air Jordan 5",
    "Air Jordan 6", "Air Jordan 7", "Air Jordan 8", "Air Jordan 9",
    "Air Jordan 10", "Air Jordan 11", "Air Jordan 12", "Air Jordan 13",
    "Air Jordan 14", "Air Jordan 15", "Air Jordan 16", "Air Jordan 17",
    "Air Jordan 18", "Air Jordan Retro 1 Off-White", "Air Jordan Retro 3 Black Cement",
    "Air Jordan Retro 4 Bred", "Air Jordan Retro 5 Grape", "Air Jordan Retro 6 Infrared",
    "Air Jordan Retro 11 Concord", "Air Jordan Retro 12 Flu Game", "Air Jordan Retro 13 He Got Game",
    "Air Jordan Retro 14 Last Shot", "Air Jordan 19 Flint"
]

print(f"\n✅ Clases definidas: {len(classes)} modelos de tenis Jordan.")

# ======================================================
# 6 Clasificar una imagen dentro de las 31 clases
# ======================================================
def clasificar_imagen(image_path):
    """
    Identifica a qué modelo Jordan pertenece la imagen dada
    comparando embeddings de texto e imagen.
    """
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(classes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # Normalización
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calcular similitudes
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_class_idx = similarities.argmax().item()

    print(f"\n🔍 Clase detectada: {classes[best_class_idx]}")
    print(f"Confianza: {similarities[0, best_class_idx].item():.2f}")
    return classes[best_class_idx], similarities[0, best_class_idx].item()

# ======================================================
# 7 Función para generar descripción de la imagen
# ======================================================
def describir_imagen(image_path):
    """
    Genera una descripción detallada de la imagen usando CLIP
    comparando con múltiples características visuales.
    """
    # Características que puede tener un tenis
    descripciones = [
        "zapatos deportivos negros", "zapatos deportivos blancos", "zapatos deportivos rojos",
        "zapatos deportivos azules", "zapatos deportivos grises", "zapatos deportivos coloridos",
        "tenis de basketball", "tenis casuales", "tenis retro", "tenis modernos",
        "zapatos con cordones blancos", "zapatos con cordones negros", "zapatos con cordones coloridos",
        "zapatos con suela blanca", "zapatos con suela negra", "zapatos con suela gruesa",
        "zapatos con diseño clásico", "zapatos con diseño moderno", "zapatos con logo visible",
        "zapatos de cuero", "zapatos de tela", "zapatos con detalles metálicos",
        "zapatos para correr", "zapatos elegantes", "zapatos cómodos",
        "zapatos con rayas", "zapatos con patrones", "zapatos lisos",
        "zapatos de perfil bajo", "zapatos de perfil alto", "zapatos acolchados"
    ]
    
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(descripciones).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        # Normalización
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calcular similitudes
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Obtener las 3 descripciones más probables
        top_similarities, top_indices = similarities[0].topk(3)
        
    descripcion = "📝 **Descripción de la imagen:**\n"
    for i, (sim, idx) in enumerate(zip(top_similarities, top_indices)):
        confidence = sim.item() * 100
        descripcion += f"   • {descripciones[idx]} ({confidence:.1f}% confianza)\n"
    
    return descripcion

# ======================================================
# 8 Obtener recomendaciones visuales basadas en similitud
# ======================================================
def recomendar(imagen_consulta, top_k=5, use_augmentation=True):
    """
    Muestra las imágenes más parecidas del dataset
    según embeddings CLIP y similitud coseno.
    """
    # Clasificación de la imagen
    clase_detectada, score = clasificar_imagen(imagen_consulta)
    
    # Embedding de la consulta
    query_emb = get_embedding(imagen_consulta, use_augmentation=use_augmentation)
    similitudes = (embeddings @ query_emb.T).squeeze(1)
    valores, indices = similitudes.topk(top_k)
    
    # Mostrar texto
    print(f"\n Recomendaciones más parecidas (con augmentation):")
    for i in indices:
        print(f"{image_paths[i]}  (Clase: {labels[i]})")

    # Mostrar visualmente
    fig, axes = plt.subplots(1, top_k+1, figsize=(15, 5))
    axes[0].imshow(Image.open(imagen_consulta))
    axes[0].set_title(f"Consulta\n({clase_detectada})")
    axes[0].axis("off")

    for idx, i in enumerate(indices):
        axes[idx+1].imshow(Image.open(image_paths[i]))
        axes[idx+1].set_title(f"Recomendación {idx+1}\n{labels[i]}")
        axes[idx+1].axis("off")

    plt.show()
    
    # Generar y mostrar descripción de la imagen
    descripcion = describir_imagen(imagen_consulta)
    print(f"\n{descripcion}")
    
    return indices, valores

# ======================================================
# 9 Ejemplo de uso (imagen de entrada)
# ======================================================
# Puedes cambiar "prueba.webp" por cualquier imagen del catálogo
recomendar("prueba.webp", top_k=5)

# ======================================================
# 🔟 Conclusiones (para la rúbrica)
# ======================================================
print("\n📘 CONCLUSIONES:")
pprint("""
El uso de CLIP demuestra el potencial de los modelos multimodales para unir comprensión visual y semántica textual sin necesidad de entrenamiento supervisado específico para cada clase. Esto permite ampliar fácilmente el sistema a nuevos modelos de tenis simplemente añadiendo descripciones textuales, sin requerir retraining completo.

Gracias a la representación embebida compartida de CLIP, el sistema puede incorporar nuevas imágenes o categorías sin cambios estructurales. Esto lo hace ideal para catálogos en crecimiento o adaaptaciones a nuestra aplicación comerciales donde se agregan productos frecuentemente, manteniendo bajos costos de mantenimiento y entrenamiento. De igual forma, nos permite expandir la tiendaa y no quedarnos en ropa solaamente. Permitiendo tener nuevos usuarios y entrar más de lleno en el mundo del retail.
""")
