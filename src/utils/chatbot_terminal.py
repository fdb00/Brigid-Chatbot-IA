import os
import random
import json
import pickle
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import google.generativeai as genai

# Variabile globale per memorizzare la cronologia
chat_history = []

# Configura Gemini
def configure_gemini():
    api_key = os.getenv('SECRET_API_KEY')
    genai.configure(api_key=api_key)

# Inizializza il modello BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)  # Aggiorna num_labels!

# Carica intents e modelli ausiliari
with open('../data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

classes = pickle.load(open('../models/classes.pkl', 'rb'))

# Funzione per ottenere la previsione di BERT
def predict_intent(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=50)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    max_prob = np.max(probs)
    predicted_class = np.argmax(probs)
    return predicted_class, max_prob

# Funzione per ottenere la risposta da Gemini
def get_gemini_response(input_text):
    global chat_history

    # Configura il modello Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    convo = model.start_chat(history=chat_history)

    # Genera una risposta usando Gemini
    response = convo.send_message(input_text)

    return response.text

# Pipeline per rispondere ai messaggi
def get_response_from_pipeline(input_text):
    global chat_history
    predicted_class, confidence = predict_intent(input_text)
    if confidence > 0.75:  # Threshold per BERT
        intent_tag = classes[predicted_class]
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])

                # Aggiungi la risposta del chatbot alla cronologia
                chat_history.append({"role": "assistant", "content": response})

                return response
    else:
        return get_gemini_response(input_text)

# Interfaccia utente della chatbot
print("Hi there! I'm Brigid, and I'm here to help you!")
while True:
    user_input = input("> ")
    # Aggiungi il messaggio dell'utente alla cronologia
    chat_history.append({"role": "user", "content": user_input})

    response = get_response_from_pipeline(user_input)
    print(response)
