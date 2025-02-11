from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
import json
import pickle
import numpy as np
import torch
import contractions
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
import google.generativeai as genai
from nrclex import NRCLex
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Abilita CORS per tutte le rotte

# Variabile globale per la cronologia della conversazione
chat_history = {"contents": []}

# Configura Gemini
api_key = os.getenv('SECRET_API_KEY')
genai.configure(api_key=api_key)

# Carica il tokenizer e il modello di BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Carica intents e modelli ausiliari
with open('../data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

classes = pickle.load(open('../models/classes.pkl', 'rb'))
model = load_model('../models/chatbot_model.keras')

# Funzione per preprocessare la frase
def preprocess_sentence(sentence):
    # Rimuovere ripetizioni di caratteri
    sentence = re.sub(r"(.)\1{2,}", r"\1", sentence)  # Esempio: "heeeelp" -> "help"

    # Correggi l'input dell'utente
    corrected_sentence = str(TextBlob(sentence).correct())

    # Espandi le contrazioni
    expanded_sentence = contractions.fix(corrected_sentence)

    # Rilevamento delle emozioni
    def detect_emotions(text):
        nrc_analysis = NRCLex(text)
        return nrc_analysis.affect_frequencies

    emotions = detect_emotions(expanded_sentence)

    # Tokenizza e ottieni gli embedding di BERT
    inputs = tokenizer(expanded_sentence, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return np.expand_dims(embedding, axis=0)

# Funzione per ottenere la previsione del modello
def predict_intent(sentence):
    bag = preprocess_sentence(sentence)
    res = model.predict(bag)[0]
    max_prob = np.max(res)
    predicted_class = np.argmax(res)
    return predicted_class, max_prob

# Funzione per ottenere la risposta da Gemini
def get_gemini_response(input_text):
    global chat_history
    model = genai.GenerativeModel("gemini-1.5-flash")
    convo = model.start_chat(history=chat_history["contents"])
    response = convo.send_message("You are Brigid, a chatbot for psychological support. Generate a response of max 200 characters for:" + input_text)
    chat_history["contents"].append({"role": "model", "parts": [{"text": response.text}]})
    return response.text

# Pipeline per rispondere ai messaggi
def get_response_from_pipeline(input_text):
    global chat_history
    predicted_class, confidence = predict_intent(input_text)
    if confidence > 0.2:
        intent_tag = classes[predicted_class]
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                chat_history["contents"].append({"role": "model", "parts": [{"text": response}]})
                return response
    else:
        return get_gemini_response(input_text)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({"error": "Missing 'message' in request"}), 400
    chat_history["contents"].append({"role": "user", "parts": [{"text": user_input}]})
    response = get_response_from_pipeline(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
