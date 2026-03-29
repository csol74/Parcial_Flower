# app.py - Aplicación Streamlit con Cámara y URL
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(page_title="Clasificador de Flores", page_icon="🌸", layout="wide")

# Título principal
st.title("🌼 Clasificador de Tipos de Flores 🌻")
st.markdown("---")

# Definir clases
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
CLASS_NAMES_ES = {
    'daisy': 'Margarita',
    'dandelion': 'Diente de León',
    'roses': 'Rosa',
    'sunflowers': 'Girasol',
    'tulips': 'Tulipán'
}

# Cargar modelo

@st.cache_resource
def load_model():
    try:
        import keras
        keras.config.enable_unsafe_deserialization()  

        model = tf.keras.models.load_model(
            'flower_model.keras',
            compile=False  
        )
        return model

    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.info("Asegúrate de que el archivo flower_model.keras esté en la carpeta")
        return None

# Función de preprocesamiento
def preprocess_image(image):
    # Redimensionar a 150x150
    image = image.resize((150, 150))
    # Convertir a array y normalizar
    img_array = np.array(image) / 255.0
    # Añadir dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Función de predicción
def predict_image(model, image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class = CLASSES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predictions[0], predicted_class, confidence

# Función para cargar imagen desde URL
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error al cargar imagen desde URL: {e}")
        return None

# Sidebar para información
with st.sidebar:
    st.header("📋 Información")
    st.write("**Clases disponibles:**")
    for class_name in CLASSES:
        st.write(f"- {CLASS_NAMES_ES[class_name]} ({class_name})")
    
    st.write("---")
    st.write("**Instrucciones:**")
    st.write("1. Elige una opción: Cargar archivo, usar cámara o URL")
    st.write("2. Espera a que el modelo procese")
    st.write("3. Observa los resultados")
    
    st.write("---")
    st.write("**Tipos de flores soportados:**")
    st.write("- 🌼 Margarita (Daisy)")
    st.write("- 🌿 Diente de León (Dandelion)")
    st.write("- 🌹 Rosa (Rose)")
    st.write("- 🌻 Girasol (Sunflower)")
    st.write("- 🌷 Tulipán (Tulip)")

# Área principal - Selector de método de entrada
st.subheader("📷 Selecciona método de entrada")
input_method = st.radio(
    "¿Cómo quieres cargar la imagen?",
    ["📁 Subir archivo", "📸 Usar cámara", "🔗 URL de imagen"],
    horizontal=True
)

# Variable para almacenar la imagen y resultados
image = None
probabilities = None
predicted_class = None
confidence = None

# Opción 1: Subir archivo
if input_method == "📁 Subir archivo":
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de flor...",
        type=['jpg', 'jpeg', 'png'],
        help="Puedes subir imágenes en formato JPG, JPEG o PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# Opción 2: Usar cámara
elif input_method == "📸 Usar cámara":
    st.info("📸 Activa tu cámara para tomar una foto")
    camera_image = st.camera_input("Tomar foto")
    
    if camera_image is not None:
        image = Image.open(camera_image)

# Opción 3: URL de imagen
elif input_method == "🔗 URL de imagen":
    url = st.text_input("Ingresa la URL de la imagen:", placeholder="https://ejemplo.com/flor.jpg")
    
    if url:
        with st.spinner('Cargando imagen desde URL...'):
            image = load_image_from_url(url)

# Procesar imagen si existe
if image is not None:
    # Crear dos columnas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Imagen cargada:")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🔍 Resultados")
        
        with st.spinner('Analizando imagen...'):
            # Cargar modelo
            model = load_model()
            
            if model is not None:
                # Realizar predicción
                probabilities, predicted_class, confidence = predict_image(model, image)
                
                # Mostrar resultado principal
                st.success(f"### 🌟 Predicción: {CLASS_NAMES_ES[predicted_class]}")
                st.info(f"**Confianza:** {confidence:.2f}%")
                
                # Barra de progreso visual
                st.progress(confidence / 100)
            else:
                st.error("No se pudo cargar el modelo. Verifica que el archivo flower_model.keras existe.")
    
    # Solo mostrar gráficos si tenemos predicciones
    if probabilities is not None:
        # Gráfico de probabilidades (debajo de las columnas)
        st.subheader("📊 Distribución de probabilidades:")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#4CAF50' if i == np.argmax(probabilities) else '#ddd' 
                 for i in range(len(CLASSES))]
        
        bars = ax.bar(range(len(CLASSES)), probabilities * 100, color=colors)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([CLASS_NAMES_ES[c] for c in CLASSES], rotation=45, ha='right')
        ax.set_ylabel('Probabilidad (%)')
        ax.set_title('Distribución de Probabilidades por Clase')
        ax.set_ylim([0, 100])
        
        # Agregar valores en las barras
        for bar, prob in zip(bars, probabilities * 100):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Mostrar tabla de probabilidades
        with st.expander("📋 Ver probabilidades detalladas"):
            prob_df = pd.DataFrame({
                'Clase': [CLASS_NAMES_ES[c] for c in CLASSES],
                'Nombre Científico': CLASSES,
                'Probabilidad (%)': probabilities * 100
            })
            prob_df = prob_df.sort_values('Probabilidad (%)', ascending=False)
            st.dataframe(prob_df.style.format({'Probabilidad (%)': '{:.2f}'}))
            
            # Mostrar barra horizontal para cada clase
            st.markdown("### Visualización por clase:")
            for i, (name_es, name_en) in enumerate(zip(CLASS_NAMES_ES.values(), CLASSES)):
                prob = probabilities[i] * 100
                st.write(f"**{name_es}** ({name_en})")
                st.progress(prob / 100, text=f"{prob:.1f}%")

else:
    # Mensaje inicial según método seleccionado
    if input_method == "📁 Subir archivo":
        st.info("👈 Sube una imagen para comenzar el análisis")
    elif input_method == "📸 Usar cámara":
        st.info("👈 Activa la cámara y toma una foto para comenzar")
    elif input_method == "🔗 URL de imagen":
        st.info("👈 Ingresa una URL de imagen para comenzar")
    
    st.markdown("---")
    st.markdown("### 🌸 Ejemplos de imágenes esperadas:")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("🌼 **Margarita**")
        st.caption("Pétalos blancos, centro amarillo")
    with col2:
        st.markdown("🌿 **Diente de León**")
        st.caption("Flores amarillas, múltiples pétalos")
    with col3:
        st.markdown("🌹 **Rosa**")
        st.caption("Pétalos enrollados, roja/rosa")
    with col4:
        st.markdown("🌻 **Girasol**")
        st.caption("Grande, pétalos amarillos")
    with col5:
        st.markdown("🌷 **Tulipán**")
        st.caption("Forma de campana")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Modelo CNN desarrollado para clasificación de flores | Entrenado con Flower Photos Dataset</p>
    <p>⚠️ Para mejor precisión, asegúrate de que la imagen esté bien iluminada y la flor sea claramente visible</p>
    <p>📷 Puedes usar: Subir archivo | Cámara en tiempo real | URL de imagen</p>
</div>
""", unsafe_allow_html=True)