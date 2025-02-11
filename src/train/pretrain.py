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
import seaborn as sns

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

def generate_training_report(model, train_x, train_y, history, intents_to_include, output_file="training_report.html"):
    """
    Genera un file HTML con le metriche e i grafici del training.
    """
    def plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return base64_img

    # 1. Grafico delle metriche di base (Loss e Accuracy)
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
    f1 = report['weighted avg']['f1-score']

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

    # 4. Confusion Matrix for Specific Intents
    global classes

    # Use the provided list of intents
    top_intents = intents_to_include
    top_indices = [classes.index(intent) for intent in top_intents if
                   intent in classes]  # Handle cases where intent might not be in the list

    if not top_indices:
        print("Warning: None of the specified intents were found in the training data.")
        return  # Exit early if no valid intents are found

    # Filter predictions and true labels to include only the specified intents
    mask = np.isin(np.argmax(train_y, axis=1), top_indices)  # Use np.argmax to get indices of true labels
    y_true_top = np.argmax(train_y, axis=1)[mask]
    y_pred_top = np.argmax(model.predict(train_x), axis=1)[
        mask]  # Predict on the train_x and get the indices of the predicted labels

    # Compute the confusion matrix for the specified intents
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_indices, normalize='true')

    # Compute the confusion matrix for top-N intents
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_indices, normalize='true')  # top_indices are still needed for labels

    # Create the confusion matrix plot
    fig_cm, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=top_intents, yticklabels=top_intents, cmap="Blues", ax=ax)
    # Add labels to the confusion matrix plot
    ax.set(
        xlabel='Predicted Label (X-axis)',  # X-axis label
        ylabel='True Label (Y-axis)',      # Y-axis label
        title=f'Confusion Matrix (For Specified Intents, Normalized)'
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    confusion_img = plot_to_base64(fig_cm)
    plt.close(fig_cm)

    # 5. Tempo medio di inferenza
    start_time = time.time()
    for _ in range(100):  # Inferenze ripetute per avere una stima affidabile
        model.predict(train_x[:10])
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / (100 * 10)

    # 6. Overfitting e Underfitting
    overfitting_analysis = ""
    if 'val_loss' in history:
        train_loss_final = history['loss'][-1]
        val_loss_final = history['val_loss'][-1]
        overfitting_analysis = (
            f"La differenza finale tra la perdita di training ({train_loss_final:.4f}) "
            f"e validation ({val_loss_final:.4f}) suggerisce "
            f"{'overfitting' if val_loss_final > train_loss_final else 'underfitting'}."
        )

    # Struttura HTML del report
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
        <p><strong>F1-Score:</strong> {f1:.4f}</p>

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

# Define the intents you want to include in the confusion matrix
mental_health_intents = [
    "mental_health",
    "anxiety_and_panic_attacks",
    "depression_and_loneliness",
    "about",
    "greeting",
    "grief_and_loss",
    "workplace_stress",
    "lgbtq_mental_health",
    "bullying",
    "victim_panic_attacks",
    "therapy_meaning"
]

# Chiamata alla funzione per generare il report
generate_training_report(
    model=model,
    train_x=train_x,
    train_y=train_y,
    history=hist.history,
    intents_to_include=mental_health_intents,  # Pass the list of intents
    output_file="training_report.html"
)
