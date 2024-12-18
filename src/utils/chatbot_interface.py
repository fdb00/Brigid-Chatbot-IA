import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import spacy

# Carica il modello spaCy
nlp = spacy.load('en_core_web_sm')

# Funzione per caricare il dataset
def load_dataset(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    else:
        print(f"Il file {file_path} non esiste!")
        return None

# Percorsi ai file dei dataset
file_path_1 = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
file_path_2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents_2.json')
file_path_3 = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents_3.json')

# Carica i modelli e i dati necessari
words = pickle.load(open('../models/words.pkl', 'rb'))
classes = pickle.load(open('../models/classes.pkl', 'rb'))
model = tf.keras.models.load_model('../models/combined_chatbot_model.h5')

# Funzione di preprocessing del testo
def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

# Funzione per fare la previsione della risposta
def predict_class(sentence):
    # Ottieni la "bag of words"
    bow_input = bow(sentence, words)
    prediction = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.05
    predicted_class_index = np.argmax(prediction)
    probability = prediction[predicted_class_index]

    if probability > ERROR_THRESHOLD:
        return classes[predicted_class_index]
    else:
        return None

# Funzione per creare la "bag of words"
def bow(sentence, words):
    sentence_words = preprocess_text(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Funzione per ottenere la risposta dalla classe prevista
def get_response(intent):
    intents_data = load_dataset(file_path_1)
    intents_2_data = load_dataset(file_path_2)
    intents_3_data = load_dataset(file_path_3)

    if intents_data:
        for intent_data in intents_data['intents']:
            if intent_data['tag'] == intent:
                return random.choice(intent_data['responses'])

    if intents_2_data:
        for context_data in intents_2_data['intents']:
            if context_data.get('tag') == intent:
                return random.choice(context_data.get('responses', ["Non ho una risposta per questo contesto."]))

    if intents_3_data:
        for intent_data in intents_3_data['intents']:
            if intent_data['tag'] == intent:
                return random.choice(intent_data['responses'])

    return "Sorry, I didn't understand that. Could you please try another way?"

# Interfaccia a riga di comando per il chatbot
def chat():
    print("Chatbot: Thank you for using Brigid! Feel free to talk to me about anything, and if you want to stop this conversation, type 'exit'")

    while True:
        # Acquisisci l'input dell'utente
        message = input("You: ")

        # Se l'utente scrive 'exit', esce dal loop
        if message.lower() == 'exit':
            print("Chatbot: Bye-bye!")
            break

        # Prevedi la classe dell'input
        predicted_class = predict_class(message)

        if predicted_class:
            # Ottieni la risposta in base alla classe prevista
            response = get_response(predicted_class)
            print(f"Chatbot: {response}")
        else:
            print("Chatbot: I'm sorry, I couldn't understand that. Could you please try again?")

# Avvia il chatbot
if __name__ == "__main__":
    chat()