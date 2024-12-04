import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import spacy

# Assicurati di avere spaCy e il modello "en_core_web_sm" installato
# Installazione: python -m pip install spacy
# Modello: python -m spacy download en_core_web_sm

# Carica il modello spaCy
nlp = spacy.load('en_core_web_sm')

# Funzione per caricare dataset
def load_dataset(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Il file {file_path} è stato caricato!")
            return data
    else:
        print(f"Il file {file_path} non esiste!")
        return None

# Percorsi ai file dei dataset
file_path_1 = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
file_path_2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents_2.json')

# Carica i dataset
intents_1 = load_dataset(file_path_1)
intents_2 = load_dataset(file_path_2)

if intents_1 is None or intents_2 is None:
    print("Errore: uno dei file non è stato trovato!")
    exit()

# Inizializza contenitori
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', '\n']

# Processa il primo dataset
for intent in intents_1['intents']:
    for pattern in intent['patterns']:
        doc = nlp(pattern)
        word_list = [token.lemma_.lower() for token in doc if token.text not in ignore_letters]
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Processa il secondo dataset
for item in intents_2:
    context = item['Context']
    response = item['Response']
    context_doc = nlp(context)
    response_doc = nlp(response)

    # Combina parole da contesto e risposta
    context_words = [token.lemma_.lower() for token in context_doc if token.text not in ignore_letters]
    response_words = [token.lemma_.lower() for token in response_doc if token.text not in ignore_letters]
    all_words = context_words + response_words

    words.extend(all_words)
    documents.append((all_words, context))  # Usa il contesto come "classe"
    if context not in classes:
        classes.append(context)

# Ordina le parole uniche e le classi
words = sorted(set(words))
classes = sorted(set(classes))

# Salva le parole e le classi
pickle.dump(words, open('../models/words.pkl', 'wb'))
pickle.dump(classes, open('../models/classes.pkl', 'wb'))

# Crea il set di dati di allenamento
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]

    # Crea una "bag of words"
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Mescola i dati e li converte in numpy array
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Crea il modello di rete neurale
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compila il modello
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Allena il modello
model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Salva il modello addestrato
model.save('combined_chatbot_model.h5')
print('Modello combinato addestrato e salvato con successo!')
