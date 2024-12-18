import spacy
import numpy as np
import pickle
import tensorflow as tf

# Carica il modello TensorFlow addestrato
model = tf.keras.models.load_model('combined_chatbot_model.h5')

# Carica le parole e le classi salvate
words = pickle.load(open('../models/words.pkl', 'rb'))
classes = pickle.load(open('../models/classes.pkl', 'rb'))

# Carica il modello spaCy
nlp = spacy.load('en_core_web_sm')

# Funzione per pre-processare e lemmatizzare la frase dell'utente
def clean_up_sentence(sentence):
    sentence = sentence.lower()
    doc = nlp(sentence)
    return [token.lemma_ for token in doc if token.text not in ['?', '!', '.', ',', '\n']]

# Funzione per creare una "bag of words"
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Funzione per predire la classe
def predict_class(sentence):
    bow_input = bow(sentence, words)
    prediction = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(prediction) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": r[1]} for r in results]

# Funzione per rispondere
def get_response(intent):
    responses = {
        "greeting": "Ciao! Come posso aiutarti oggi?",
        "goodbye": "Arrivederci!",
        "thanks": "Grazie a te!"
    }
    return responses.get(intent, "Mi dispiace, non ho capito la tua domanda.")

# Interfaccia del chatbot
def chatbot():
    print("Chatbot: Ciao! Sono qui per aiutarti.")
    while True:
        message = input("Tu: ")
        if message.lower() == "exit":
            print("Chatbot: Arrivederci!")
            break
        intents = predict_class(message)
        response = get_response(intents[0]["intent"])
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
