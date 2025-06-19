import requests
import torch 
import torch.nn as nn
from transformers import  DistilBertModel
import time
import urllib.parse as up

def obtener_tweets_prueba(username, max_results=30):
    """
    Función de prueba que devuelve tweets simulados para desarrollo
    """
    tweets_prueba = [
        "¡Qué día tan maravilloso! Me siento muy feliz y agradecido por todo lo que tengo.",
        "Estoy un poco preocupado por el examen de mañana, pero confío en que todo saldrá bien.",
        "Me siento muy frustrado con este proyecto, parece que nada sale como esperaba.",
        "¡Increíble! Acabo de recibir una oferta de trabajo que no puedo creer.",
        "Hoy me siento un poco triste, pero sé que mañana será mejor.",
        "Estoy muy emocionado por el concierto de esta noche, será increíble.",
        "Me siento un poco abrumado con tantas tareas pendientes.",
        "¡Qué sorpresa tan agradable! No me lo esperaba para nada.",
        "Estoy muy relajado después de una larga semana de trabajo.",
        "Me siento muy confiado sobre el futuro, todo parece alinearse perfectamente."
    ]
    return tweets_prueba[:max_results]

def obtener_tweets(username, max_results=30):
    BEARER_TOKEN = up.unquote("AAAAAAAAAAAAAAAAAAAAADVX2gEAAAAAfzaERTxvbmMm0rBss8jqswi8iZU%3DP8g6DGDdMHp1RazL2Z2vVF4PH1rtug6mQ8No0Gq1dV0YVQCW4Q")

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }

    # Obtener el ID del usuario por nombre de usuario
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        print(f"Error al obtener ID: {resp.status_code} - {resp.text}")
        return obtener_tweets_prueba(username)

    user_id = resp.json()["data"]["id"]

    # Obtener los tweets del usuario
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        "max_results": max_results,
        "tweet.fields": "text"
    }

    resp = requests.get(url, headers=headers, params=params)

    if resp.status_code != 200:
        print(f"Error al obtener tweets: {resp.status_code} - {resp.text}")
        return obtener_tweets_prueba(username)

    tweets = resp.json().get("data", [])
    return [tweet["text"] for tweet in tweets]

class BertEmotionClassifier(nn.Module): 
    def __init__(self, num_labels=20): 
        super(BertEmotionClassifier, self).__init__() 
        self.bert = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased") 
        self.dropout = nn.Dropout(0.3) 
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels) 
        self.softmax = nn.Softmax(dim=1) 
    def forward(self, input_ids, attention_mask): 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) 
        cls_output = outputs.last_hidden_state[:, 0, :] 
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        probs = self.softmax(logits)
        return probs