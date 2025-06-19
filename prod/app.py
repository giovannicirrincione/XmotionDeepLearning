import streamlit as st
import torch
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import utils_dev as utils_dev
from utils_dev import obtener_tweets
from utils_dev import BertEmotionClassifier
import pickle
import os
import warnings
import re
import emoji

# Suprimir advertencias de scikit-learn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configuración de la página
st.set_page_config(page_title="Análisis de Emociones en Tweets", layout="wide")

# Configurar rutas de archivos
PROD_DIR = os.path.dirname(os.path.abspath(__file__))

# Tu modelo y clases ya definidos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

# Cargar encoder
with open(os.path.join(PROD_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

# Cargar traducción
with open(os.path.join(PROD_DIR, "translation.pkl"), "rb") as f:
    translation = pickle.load(f)

# Cargar modelo entrenado
@st.cache_resource
def cargar_modelo():
    model = BertEmotionClassifier(num_labels=20).to(device)
    model.load_state_dict(torch.load(os.path.join(PROD_DIR, "modelo_lr_2e-05.pth"), map_location=device))
    model.eval()
    return model

model = cargar_modelo()

# Función para limpiar texto
def clean_text(text):
    # Pasar a minúsculas
    text = text.lower()

    # Eliminar links
    text = re.sub(r"http\S+|www\S+|pic\.twitter\.com\S+", "", text)

    # Eliminar menciones y hashtags
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # Eliminar emojis
    text = emoji.replace_emoji(text, replace="")

    # Eliminar caracteres especiales, excepto letras y espacios
    text = re.sub(r"[^\w\s]", "", text)

    # Eliminar múltiples espacios
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# Función para predecir emociones de múltiples textos
def analizar_emociones_tweets(lista_tweets):
    if not lista_tweets:
        return {}
        
    # Inicializamos un acumulador para las emociones
    suma_emociones = {}
    for emocion in encoder.classes_:
        suma_emociones[emocion] = 0.0

    for oracion in lista_tweets:
        texto_limpio = clean_text(oracion)
        tokens = tokenizer(texto_limpio, return_tensors="pt", padding=True, truncation=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            probs = model(**tokens)[0]

        for i in range(len(probs)):
            emocion = encoder.classes_[i]
            suma_emociones[emocion] += float(probs[i] * 100)

    # Promediamos dividiendo por la cantidad de tweets
    cantidad = len(lista_tweets)
    promedio_emociones = {emocion: valor / cantidad for emocion, valor in suma_emociones.items()}

    # Ordenamos de mayor a menor
    return dict(sorted(promedio_emociones.items(), key=lambda x: -x[1]))

# Datos de prueba para cuando la API está limitada
TWEETS_PRUEBA = [
    "¡Qué día tan maravilloso! Me siento muy feliz y agradecido por todo lo que tengo.",
    "Estoy realmente emocionado por el nuevo proyecto que estamos comenzando.",
    "Hoy me siento un poco triste, pero sé que mañana será mejor.",
    "Me siento muy enojado por la situación actual, necesitamos un cambio.",
    "Estoy sorprendido por las noticias de hoy, no me lo esperaba.",
    "Me siento muy ansioso por la presentación de mañana.",
    "Qué alegría ver a todos reunidos después de tanto tiempo.",
    "Me siento decepcionado por los resultados obtenidos.",
    "Estoy muy orgulloso de lo que hemos logrado juntos.",
    "Me siento confundido con todas estas nuevas reglas."
]

# Interfaz de Streamlit
st.title("Análisis de emociones en tweets")
st.markdown("""
Esta aplicación analiza los tweets de un usuario de Twitter y determina las emociones predominantes en sus publicaciones.
""")

usuario = st.text_input("Ingresá el nombre de usuario de Twitter (sin el @)")

if st.button("Analizar tweets") and usuario:
    with st.spinner('Obteniendo tweets...'):
        tweets = []
        usando_datos_prueba = False
        
        try:
            tweets = obtener_tweets(usuario)
            if not tweets:  # Si no se obtuvieron tweets
                raise Exception("No se pudieron obtener tweets")
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "UsageCapExceeded" in error_msg or "Too Many Requests" in error_msg:
                st.warning("""
                ⚠️ La API de Twitter está temporalmente limitada. 
                Se utilizarán datos de prueba para demostrar la funcionalidad.
                """)
                tweets = TWEETS_PRUEBA
                usando_datos_prueba = True
            else:
                st.error(f"""
                Error al obtener tweets: {error_msg}
                
                Por favor, intente nuevamente más tarde o con otro usuario.
                """)
        
        if tweets:
            st.success(f"Se obtuvieron {len(tweets)} tweets")
            
            with st.spinner('Analizando emociones...'):
                emociones = analizar_emociones_tweets(tweets)

                # Mostrar resultados
                st.subheader("Resultados del análisis")
                
                # Obtener los 5 estados emocionales con mayor porcentaje
                df_emociones = pd.DataFrame(list(emociones.items()), columns=["Emoción", "Porcentaje"])
                df_top5 = df_emociones.head(5)
                
                # Gráfico de barras horizontal
                st.subheader("Top 5 emociones predominantes")
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(df_top5["Emoción"], df_top5["Porcentaje"], height=0.5)
                ax.set_xlabel("Porcentaje (%)")
                ax.set_xlim(0, 100)
                
                # Agregar los valores de porcentaje al final de cada barra
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                           f'{width:.1f}%', ha='left', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)

                # Gráfico de torta
                st.subheader("Distribución porcentual (Top 5)")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(df_top5["Porcentaje"], labels=df_top5["Emoción"], 
                      autopct='%1.1f%%', startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

                # Mostrar tweets analizados
                st.subheader("Tweets analizados")
                for tweet in tweets:
                    st.write(f"- {clean_text(tweet)}")
                    
                if usando_datos_prueba:
                    st.info("""
                    ℹ️ Nota: Estos son datos de prueba utilizados debido a limitaciones de la API de Twitter.
                    Los resultados mostrados son solo para fines de demostración.
                    """)