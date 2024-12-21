import random
import json
import pickle
import numpy as np

import nltk
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('../data/intents.json', encoding='utf-8').read())

words = pickle.load(open('../models/words.pkl', 'rb'))
classes = pickle.load(open('../models/classes.pkl', 'rb'))
model = load_model('../models/chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.23
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    if not results:
        return []  # Nessuna predizione valida
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:  # Se intents_list è vuoto
        return "Sorry, I didn't understand that. Could you please try again?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def parseDict2String(intents_list): #modifica che servirà per la raccolta dati
    intentStr = str(intents_list[0])
    print(type(intentStr) is str)

print("Hi there! I'm Brigid, and I'm here to help you!")
while True:
    message = input("> ")
    ints = predict_class(message)
    res = get_response(ints, intents) #ints contiene anche il tag della conversazione
    print(res)
   # parseDict2String(ints)



