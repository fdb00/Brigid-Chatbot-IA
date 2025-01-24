import numpy as np
import json
import pickle
from transformers import BertTokenizer, BertModel
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import io
import base64
import time

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


def generate_training_report(model, train_x, train_y, history, output_file="training_report.html"):
    """
    Genera un file HTML con le metriche e i grafici del training.

    :param model: Modello Keras addestrato.
    :param train_x: Dati di input per il training.
    :param train_y: Etichette di output (one-hot encoded) del training.
    :param history: Cronologia del training (es. hist.history).
    :param output_file: Nome del file HTML da generare.
    """

    # Funzione per convertire un grafico in immagine base64
    def plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return base64_img

    # 1. Grafico delle metriche di base
    fig_loss = plt.figure(figsize=(8, 4))
    plt.plot(history['loss'], label='Loss', color='blue')
    plt.plot(history.get('val_loss', []), label='Val Loss', color='red', linestyle='dashed')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    loss_img = plot_to_base64(fig_loss)
    plt.close(fig_loss)

    fig_accuracy = plt.figure(figsize=(8, 4))
    plt.plot(history['accuracy'], label='Accuracy', color='green')
    plt.plot(history.get('val_accuracy', []), label='Val Accuracy', color='orange', linestyle='dashed')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    accuracy_img = plot_to_base64(fig_accuracy)
    plt.close(fig_accuracy)

    # 2. Metriche aggiuntive (Precision, Recall, F1-Score)
    y_pred = model.predict(train_x)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(train_y, axis=1)
    report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    # 3. Confidence Distribution
    confidences = np.max(y_pred, axis=1)
    fig_confidence = plt.figure(figsize=(8, 4))
    plt.hist(confidences, bins=20, color='purple', alpha=0.7)
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid()
    confidence_img = plot_to_base64(fig_confidence)
    plt.close(fig_confidence)

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    fig_cm, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(train_y[0])))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    confusion_img = plot_to_base64(fig_cm)
    plt.close(fig_cm)

    # 5. Tempo medio di inferenza
    start_time = time.time()
    for _ in range(100):  # Inferenze ripetute per avere una stima affidabile
        model.predict(train_x[:10])
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / (100 * 10)

    # 6. Overfitting e Underfitting (differenza tra training e validation metrics)
    overfitting_analysis = ""
    if 'val_loss' in history:
        train_loss_final = history['loss'][-1]
        val_loss_final = history['val_loss'][-1]
        overfitting_analysis = f"La differenza finale tra la perdita di training ({train_loss_final:.4f}) e validation ({val_loss_final:.4f}) suggerisce {'overfitting' if val_loss_final > train_loss_final else 'underfitting'}."

    # Struttura HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Training Report</title>
    </head>
    <body>
        <h1>Training Report</h1>

        <h2>1. Metriche di Base</h2>
        <h3>Loss</h3>
        <img src="data:image/png;base64,{loss_img}" alt="Loss Graph">
        <h3>Accuracy</h3>
        <img src="data:image/png;base64,{accuracy_img}" alt="Accuracy Graph">

        <h2>2. Metriche Aggiuntive</h2>
        <p><strong>Precision:</strong> {precision:.4f}</p>
        <p><strong>Recall:</strong> {recall:.4f}</p>
        <p><strong>F1-Score:</strong> {f1_score:.4f}</p>

        <h2>3. Confidence Distribution</h2>
        <img src="data:image/png;base64,{confidence_img}" alt="Confidence Distribution">

        <h2>4. Confusion Matrix</h2>
        <img src="data:image/png;base64,{confusion_img}" alt="Confusion Matrix">

        <h2>5. Tempo Medio di Inferenza</h2>
        <p>{avg_inference_time:.6f} secondi per campione</p>

        <h2>6. Overfitting e Underfitting</h2>
        <p>{overfitting_analysis}</p>
    </body>
    </html>
    """

    # Salva il file HTML
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report HTML generato: {output_file}")


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
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)
hist = model.fit(train_x, train_y, epochs=100, batch_size=8, verbose=1)

# Salva il modello addestrato
model.save('../models/chatbot_model.keras')

y_pred = model.predict(train_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(train_y, axis=1)

f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
print(f"F1-Score: {f1}")
print("Modello addestrato con successo utilizzando BERT embeddings!")

# Chiamata alla funzione per generare il report
generate_training_report(
    model=model,
    train_x=train_x,
    train_y=train_y,
    history=hist.history,  # Passa lo storico dell'addestramento
    output_file="training_report.html"  # Nome del file HTML
)
