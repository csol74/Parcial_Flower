# app.py - Aplicación Streamlit
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

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
    model = tf.keras.models.load_model('flower_classifier_final.h5')
    return model

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

# Sidebar para información
with st.sidebar:
    st.header("📋 Información")
    st.write("**Clases disponibles:**")
    for class_name in CLASSES:
        st.write(f"- {CLASS_NAMES_ES[class_name]} ({class_name})")
    
    st.write("---")
    st.write("**Instrucciones:**")
    st.write("1. Sube una imagen de una flor")
    st.write("2. Espera a que el modelo procese")
    st.write("3. Observa los resultados")
    
    st.write("---")
    st.write("**Tipos de flores soportados:**")
    st.write("- 🌼 Margarita (Daisy)")
    st.write("- 🌿 Diente de León (Dandelion)")
    st.write("- 🌹 Rosa (Rose)")
    st.write("- 🌻 Girasol (Sunflower)")
    st.write("- 🌷 Tulipán (Tulip)")

# Área principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Cargar Imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de flor...",
        type=['jpg', 'jpeg', 'png'],
        help="Puedes subir imágenes en formato JPG, JPEG o PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)

with col2:
    st.subheader("🔍 Resultados")
    
    if uploaded_file is not None:
        with st.spinner('Analizando imagen...'):
            # Cargar modelo
            model = load_model()
            
            # Realizar predicción
            probabilities, predicted_class, confidence = predict_image(model, image)
            
            # Mostrar resultado principal
            st.success(f"### 🌟 Predicción: {CLASS_NAMES_ES[predicted_class]}")
            st.info(f"**Confianza:** {confidence:.2f}%")
            
            # Crear gráfico de barras para probabilidades
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#ff9999' if i == np.argmax(probabilities) else '#66b3ff' 
                     for i in range(len(CLASSES))]
            
            bars = ax.bar(range(len(CLASSES)), probabilities * 100, color=colors)
            ax.set_xticks(range(len(CLASSES)))
            ax.set_xticklabels([CLASS_NAMES_ES[c] for c in CLASSES], rotation=45)
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
            st.subheader("📊 Probabilidades detalladas")
            prob_df = pd.DataFrame({
                'Clase': [CLASS_NAMES_ES[c] for c in CLASSES],
                'Nombre Científico': CLASSES,
                'Probabilidad (%)': probabilities * 100
            })
            prob_df = prob_df.sort_values('Probabilidad (%)', ascending=False)
            st.dataframe(prob_df.style.format({'Probabilidad (%)': '{:.2f}'}))
    else:
        st.info("Por favor, carga una imagen para comenzar el análisis")
        st.markdown("---")
        st.markdown("### Ejemplos de imágenes esperadas:")
        st.markdown("""
        - 🌼 Flores con pétalos blancos y centro amarillo (Margarita)
        - 🌿 Flores amarillas con múltiples pétalos (Diente de León)
        - 🌹 Flores rojas con pétalos enrollados (Rosa)
        - 🌻 Flores grandes con pétalos amarillos (Girasol)
        - 🌷 Flores con forma de campana (Tulipán)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Modelo CNN desarrollado para clasificación de flores | Entrenado con Flower Photos Dataset</p>
    <p>⚠️ Para mejor precisión, asegúrate de que la imagen esté bien iluminada y la flor sea claramente visible</p>
</div>
""", unsafe_allow_html=True)