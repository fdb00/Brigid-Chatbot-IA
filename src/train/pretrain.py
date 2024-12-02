import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import spacy

# Assicurati di avere spaCy e il modello "en_core_web_sm" installato
# Installazione:python -m pip install spacy
# Modello: python -m spacy download en_core_web_sm

# Carica il modello spaCy
nlp = spacy.load('en_core_web_sm')

# Verifica che il percorso del file esista prima di aprirlo
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        intents = json.load(f)
        print(f"Il file {file_path} esiste!")
else:
    print(f"Il file {file_path} non esiste!")

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Processa le frasi negli intenti
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Analizza il testo con spaCy
        doc = nlp(pattern)
        # Estrai le radici (lemmi) delle parole, ignorando i caratteri non desiderati
        word_list = [token.lemma_.lower() for token in doc if token.text not in ignore_letters]
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

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
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compila il modello
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Allena il modello
model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Salva il modello addestrato
model.save('chatbot_model.h5')
print('Done')
