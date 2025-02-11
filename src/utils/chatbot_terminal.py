import os
import random
import json
import pickle
import numpy as np
import torch
import contractions
from tensorflow.keras.models import load_model
import google.generativeai as genai
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
import re
from nrclex import NRCLex

# Variabile globale per memorizzare la cronologia
chat_history = {"contents": []}

# Carica il tokenizer e il modello di BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Configura Gemini
def configure_gemini():
    api_key = os.getenv('SECRET_API_KEY')
    genai.configure(api_key=api_key)

# Carica intents e modelli ausiliari
with open('../data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

classes = pickle.load(open('../models/classes.pkl', 'rb'))
model = load_model('../models/chatbot_model.keras')

# Funzione per preprocessare la frase

def preprocess_sentence(sentence):

    # Step 1: Rimuovere ripetizioni di caratteri
    sentence = re.sub(r"(.)\1{2,}", r"\1", sentence)  # Esempio: "heeeelp" -> "help"

    # Step 2: Correggi prima la frase
    corrected_sentence = str(TextBlob(sentence).correct())

    # Step 3: Espandi le contrazioni
    expanded_sentence = contractions.fix(corrected_sentence)

    # Step 4: Rilevamento delle emozioni
    def detect_emotions(text):
        nrc_analysis = NRCLex(text)
        print(f"Affect frequencies: {nrc_analysis.affect_frequencies}")
        return nrc_analysis.affect_frequencies

    emotions = detect_emotions(expanded_sentence)

    # Tokenizza la frase espansa
    inputs = tokenizer(expanded_sentence, return_tensors="pt", padding=True, truncation=True, max_length=50)

    # Calcola gli embedding di BERT
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Prendi la media degli hidden states per rappresentare la frase
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Converte l'embedding in una forma bidimensionale (1, 768)
    return np.expand_dims(embedding, axis=0)

# Funzione per ottenere la previsione del modello
def predict_intent(sentence):
    bag = preprocess_sentence(sentence)  # Preprocessa la frase
    res = model.predict(bag)[0]  # Ottieni la probabilità per ogni classe
    max_prob = np.max(res)
    predicted_class = np.argmax(res)
    return predicted_class, max_prob

# Funzione per ottenere la risposta da Gemini
def get_gemini_response(input_text):
    global chat_history

    # Configura il modello Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    convo = model.start_chat(history=chat_history["contents"])

    # Genera una risposta usando Gemini
    response = convo.send_message(
        "You are Brigid a chatbot for psychological support. Generate a response of max 200 character for:" + input_text)

    # Aggiungi la risposta di Gemini alla cronologia
    chat_history["contents"].append({
        "role": "model",
        "parts": [{"text": response.text}]
    })
    print("Generata da Gemini")
    return response.text

# Pipeline per rispondere ai messaggi
def get_response_from_pipeline(input_text):
    global chat_history
    predicted_class, confidence = predict_intent(input_text)
    print(f"Confidence: {confidence}")
    if confidence > 0.2:  # Threshold per il modello
        intent_tag = classes[predicted_class]
        print("Generato dal modello. Tag individuato: "+intent_tag)
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])

                # Aggiungi la risposta generata dal modello alla cronologia
                chat_history["contents"].append({
                    "role": "model",
                    "parts": [{"text": response}]
                })

                return response
    else:
        return get_gemini_response(input_text)

# Interfaccia utente della chatbot
print("Hi there! I'm Brigid, and I'm here to help you!")
configure_gemini()
while True:
    user_input = input("> ")
    # Aggiungi il messaggio dell'utente alla cronologia
    chat_history["contents"].append({
        "role": "user",
        "parts": [{"text": user_input}]
    })

    response = get_response_from_pipeline(user_input)
    print(response)
