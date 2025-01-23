import numpy as np
import json
import pickle
from transformers import BertTokenizer, BertModel
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Carica il tokenizer e il modello pre-addestrato di BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Carica il file intents
with open('../data/intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Liste per documenti e classi
classes = []
documents = []

# Estrazione dei pattern e dei tag
for intent in intents['intents']:
    for pattern in intent['patterns']:
        documents.append((pattern, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

classes = sorted(set(classes))

# Funzione per generare embedding con BERT
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Usa la media degli hidden states per rappresentare la frase
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Creazione dei dati di addestramento
train_x = []
train_y = []

for pattern, tag in documents:
    # Ottieni l'embedding per la frase
    embedding = get_bert_embedding(pattern)
    train_x.append(embedding)

    # One-hot encoding del tag
    output_row = [0] * len(classes)
    output_row[classes.index(tag)] = 1
    train_y.append(output_row)

train_x = np.array(train_x)
train_y = np.array(train_y)

# Salva le classi e i dati
pickle.dump(classes, open('../models/classes.pkl', 'wb'))

# Costruzione del modello di classificazione
model = Sequential()
model.add(Dense(128, input_shape=(train_x.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilazione e addestramento
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100, batch_size=8, verbose=1)

# Salva il modello addestrato
model.save('../models/chatbot_model.keras')
print("Modello addestrato con successo utilizzando BERT embeddings!")
