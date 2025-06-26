import streamlit as st
import torch
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import utils_dev as utils_dev
from utils_dev import obtener_tweets, BertEmotionClassifier
import pickle
import os
import warnings
import re
import emoji

# Suprimir advertencias de scikit-learn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Emociones en Tweets",
    page_icon="🤖", 
    layout="wide"
)

PROD_DIR = os.path.dirname(os.path.abspath(__file__))

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
    model.load_state_dict(torch.load(os.path.join(PROD_DIR, "modelo_lr_5e-05.pth"), map_location=device))
    model.eval()
    return model

model = cargar_modelo()

# Limpiar texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|pic\.twitter\.com\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Promedio emociones para varios tweets
def analizar_emociones_tweets(lista_tweets):
    if not lista_tweets:
        return {}
    suma_emociones = {emocion: 0.0 for emocion in encoder.classes_}
    for oracion in lista_tweets:
        texto_limpio = clean_text(oracion)
        tokens = tokenizer(texto_limpio, return_tensors="pt", padding=True, truncation=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            probs = model(**tokens)[0]
        for i in range(len(probs)):
            emocion = encoder.classes_[i]
            suma_emociones[emocion] += float(probs[i] * 100)
    cantidad = len(lista_tweets)
    promedio_emociones = {emocion: valor / cantidad for emocion, valor in suma_emociones.items()}
    return dict(sorted(promedio_emociones.items(), key=lambda x: -x[1]))

# Emociones para un tweet
def predecir_emociones_individuales(texto):
    texto_limpio = clean_text(texto)
    tokens = tokenizer(texto_limpio, return_tensors="pt", padding=True, truncation=True, max_length=128)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
        probs = output[0].squeeze(0)
    return {
        emocion: prob.item() * 100
        for emocion, prob in zip(encoder.classes_, probs)
    }

# Datos de prueba
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

# Estado inicial
if "tweets" not in st.session_state:
    st.session_state.tweets = []
    st.session_state.usando_datos_prueba = False

st.title("Análisis de emociones en tweets")

st.markdown("""
Esta aplicación analiza los tweets de un usuario de Twitter (X) y determina las emociones predominantes.

### 🛠️ ¿Cómo obtener tu Bearer Token?

1. Ingresá a [X Developer Platform](https://docs.x.com/x-api/introduction)
2. Navega hacia developer Portal
3. Creá un proyecto y una app
4. Elegi nivel de acceso basico
5. Dentro de tu app, andá a la sección “Keys and Tokens”
6. Hacé clic en “Regenerate” o “View” para copiar el Bearer Token

""")
bearer_token = st.text_input("🔑 Pegá tu Bearer Token de X (requerido para acceder a tweets)", type="password")
usuario = st.text_input("👤 Ingresá el nombre de usuario de Twitter (sin @)")

if st.button("📊 Analizar tweets") and usuario:
    with st.spinner('Obteniendo tweets...'):
        tweets = []
        usando_datos_prueba = False
        try:
            tweets = obtener_tweets(usuario, bearer_token)
            if not tweets:
                raise Exception("No se pudieron obtener tweets")
        except Exception as e:
            error_msg = str(e)
            st.warning(f"⚠️ Error: {error_msg}")
            st.info("Se utilizarán datos de prueba.")
            tweets = TWEETS_PRUEBA
            usando_datos_prueba = True

        if tweets:
            st.session_state.tweets = tweets
            st.session_state.usando_datos_prueba = usando_datos_prueba

# Resultados
if st.session_state.tweets:
    with st.spinner('Analizando emociones...'):
        emociones = analizar_emociones_tweets(st.session_state.tweets)
        st.subheader(" Top 5 emociones predominantes")

        df_emociones = pd.DataFrame(list(emociones.items()), columns=["Emoción", "Porcentaje"])
        df_top5 = df_emociones.head(5)

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(df_top5["Emoción"], df_top5["Porcentaje"], height=0.5)
        ax.set_xlabel("Porcentaje (%)")
        ax.set_xlim(0, 100)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader(" Distribución porcentual (Top 5)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(df_top5["Porcentaje"], labels=df_top5["Emoción"], autopct='%1.1f%%', startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        st.subheader(" Tweets analizados")
        for tweet in st.session_state.tweets:
            with st.expander(f" {clean_text(tweet)}"):
                emociones_tweet = predecir_emociones_individuales(tweet)
                df_emocion_tweet = pd.DataFrame(list(emociones_tweet.items()), columns=["Emoción", "Probabilidad (%)"])
                df_emocion_tweet = df_emocion_tweet.sort_values(by="Probabilidad (%)", ascending=False).head(5)

                fig_tweet, ax_tweet = plt.subplots(figsize=(8, 3))
                bars = ax_tweet.barh(df_emocion_tweet["Emoción"], df_emocion_tweet["Probabilidad (%)"], color='skyblue')
                ax_tweet.set_xlim(0, 100)
                for bar in bars:
                    width = bar.get_width()
                    ax_tweet.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center')
                ax_tweet.set_xlabel("Probabilidad (%)")
                plt.tight_layout()
                st.pyplot(fig_tweet)

        st.subheader("🔍 Filtrar tweets por emoción dominante")
        emocion_seleccionada = st.selectbox("Seleccioná una emoción", encoder.classes_)

        tweets_filtrados = []
        for tweet in st.session_state.tweets:
            emociones_tweet = predecir_emociones_individuales(tweet)
            top5_emociones = sorted(emociones_tweet.items(), key=lambda x: -x[1])[:5]
            top5_dict = dict(top5_emociones)

            # Agrega solo si está en el top5 Y su valor es mayor a 0.5
            if emocion_seleccionada in top5_dict and top5_dict[emocion_seleccionada] > 0.5:
                tweets_filtrados.append((tweet, emociones_tweet))

        if tweets_filtrados:
            st.markdown(f"Se encontraron **{len(tweets_filtrados)}** tweets donde la emoción **'{emocion_seleccionada}'** está entre las 5 más probables.")
            for tweet, emociones_tweet in tweets_filtrados:
                with st.expander(f" {clean_text(tweet)}"):
                    df_emocion_tweet = pd.DataFrame(list(emociones_tweet.items()), columns=["Emoción", "Probabilidad (%)"])
                    df_emocion_tweet = df_emocion_tweet.sort_values(by="Probabilidad (%)", ascending=False).head(5)

                    fig, ax = plt.subplots(figsize=(8, 3))
                    bars = ax.barh(df_emocion_tweet["Emoción"], df_emocion_tweet["Probabilidad (%)"], color='lightgreen')
                    ax.set_xlim(0, 100)
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center')
                    ax.set_xlabel("Probabilidad (%)")
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.warning(f"No se encontraron tweets donde la emoción '{emocion_seleccionada}' esté entre las 5 más probables.")

        if st.session_state.usando_datos_prueba:
            st.info("ℹ️ Nota: Estos son datos de prueba utilizados debido a errores con la API.")
