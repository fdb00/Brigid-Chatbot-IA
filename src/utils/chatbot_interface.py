import os
import numpy as np
import pickle
import tensorflow as tf
import spacy
import random
import json

# Carica il modello di spaCy
nlp = spacy.load('en_core_web_sm')

# Percorsi relativi
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Livello base del progetto
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'combined_chatbot_model.h5')
WORDS_PATH = os.path.join(BASE_DIR, 'models', 'words.pkl')
CLASSES_PATH = os.path.join(BASE_DIR, 'models', 'classes.pkl')
INTENTS_PATH = os.path.join(BASE_DIR, 'data', 'intents.json')
INTENTS_2_PATH = os.path.join(BASE_DIR, 'data', 'intents_2.json')
INTENTS_3_PATH = os.path.join(BASE_DIR, 'data', 'intents_3.json')  # Nuovo file

# Carica il modello addestrato
model = tf.keras.models.load_model(MODEL_PATH)

# Carica le parole e le classi salvate
words = pickle.load(open(WORDS_PATH, 'rb'))
classes = pickle.load(open(CLASSES_PATH, 'rb'))


# Funzione per creare la "bag of words" dall'input dell'utente
def clean_up_sentence(sentence):
    sentence_doc = nlp(sentence)
    return [token.lemma_.lower() for token in sentence_doc if not token.is_punct]


def bow(sentence, words):
    # Crea una "bag of words" per l'input dell'utente
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Funzione per fare la previsione della risposta
def predict_class(sentence):
    # Ottieni la "bag of words"
    bow_input = bow(sentence, words)
    # Previsione del modello
    prediction = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.25
    # Ottieni l'indice della classe con la previsione più alta
    predicted_class_index = np.argmax(prediction)
    probability = prediction[predicted_class_index]

    # Se la probabilità è abbastanza alta, restituisci la classe
    if probability > ERROR_THRESHOLD:
        return classes[predicted_class_index]
    else:
        return None


# Funzione per ottenere la risposta dalla classe prevista
def get_response(intent):
    # Carica i dati degli intenti
    intents_data = load_dataset(INTENTS_PATH)
    intents_2_data = load_dataset(INTENTS_2_PATH)
    intents_3_data = load_dataset(INTENTS_3_PATH)  # Carica il terzo dataset

    # Cerca nei dataset standard
    if intents_data:
        for intent_data in intents_data['intents']:
            if intent_data['tag'] == intent:
                return random.choice(intent_data['responses'])

    # Cerca nei contesti del secondo dataset
    if intents_2_data:
        for context_data in intents_2_data:
            if context_data['Context'] == intent:
                return context_data.get('Response', "Non ho una risposta per questo contesto.")

    # Cerca nei contesti del terzo dataset (intents_3.json)
    if intents_3_data:
        for intent_data in intents_3_data['intents']:
            if intent_data['tag'] == intent:
                return random.choice(intent_data['responses'])

    return "I'm sorry, I don't understand that."


# Funzione per caricare il dataset
def load_dataset(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    else:
        print(f"Il file {file_path} non esiste!")
        return None


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
