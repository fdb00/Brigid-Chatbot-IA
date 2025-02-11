import json
import pickle
import numpy as np
import random
from tensorflow.keras.models import load_model

# Caricamento del modello e delle classi
model = load_model('../src/models/chatbot_model.keras')
classes = pickle.load(open('../src/models/classes.pkl', 'rb'))

# Caricamento del dataset
with open('../src/data/intents.json', encoding='utf-8') as file:
    intents = json.load(file)


def preprocess_sentence(sentence):
    """
    Preprocessa la frase dell'utente per trasformarla in un embedding compatibile con il modello.
    """
    from transformers import BertTokenizer, BertModel
    import torch

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return np.expand_dims(embedding, axis=0)


def predict_intent(sentence):
    """
    Predice l'intento della frase data in input e restituisce il tag previsto e la confidence.
    """
    bag = preprocess_sentence(sentence)
    res = model.predict(bag)[0]
    max_prob = np.max(res)
    predicted_class = np.argmax(res)
    return classes[predicted_class], max_prob


# TEST 1: BASELINE TEST
baseline_results = []
baseline_correct = 0
baseline_total = 0
for intent in intents['intents']:
    sample_patterns = random.sample(intent['patterns'], min(3, len(intent['patterns'])))  # Seleziona 3 pattern casuali
    for pattern in sample_patterns:
        predicted_tag, confidence = predict_intent(pattern)
        correct = predicted_tag == intent['tag']
        baseline_results.append((pattern, intent['tag'], predicted_tag, confidence, correct))
        baseline_correct += correct
        baseline_total += 1
baseline_accuracy = (baseline_correct / baseline_total) * 100

# TEST 2: GENERALIZATION TEST
custom_sentences = [
    ("Hi friend!", "greeting"),
    ("I need assistance, please", "help"),
    ("What can you do?", "capabilities"),
    ("Bye, see you later!", "goodbye"),
    ("Tell me a joke", "funny"),
    ("Tell me something about yourself", "about"),
    ("I can't help but worry", "anxious"),
    ("I need to get some rest", "sleep"),
    ("I'm really scared right now", "scared"),
    ("I've lost someone I care about", "death"),
    ("You are very annoying", "hate-you"),
    ("I feel numb inside", "depressed"),
    ("I can't tolerate who I am", "hate-me"),
    ("Do you have any jokes", "jokes"),
    ("How do you work?", "creation"),
    ("You already mentioned that", "repeat"),
    ("That’s not right", "wrong"),
    ("I’m very proud of myself", "happy"),
    ("You’re not smart!", "stupid"),
    ("Where do you come from?", "location"),
    ("Could we discuss something else?", "something-else"),
    ("How can I make friends?", "friends"),
    ("May I ask you a question?", "ask"),
    ("I have an issue", "problem"),
    ("I don't know how to deal with this", "no-approach"),
    ("I'd like to learn more about this", "learn-more"),
    ("You're correct", "user-agree"),
    ("I do meditation", "user-meditation"),
    ("Could you give me your advice?", "user-advice"),
    ("What does mental health mean?", "learn-mental-health"),
    ("Could you explain what depression is?", "depression_meaning"),
    ("What is the role of a therapist?", "therapist_role"),
    ("No answer", "no-response"),
    ("I never thought I would be able to do it, but I did", "unexpected_success"),
    ("I'm going through a difficult time", "problem"),
    ("What's going on with you?", "casual"),
    ("I’m in such a good mood today!", "happy"),
    ("Do you offer any emotional support?", "skill"),
    ("Who made you?", "creation"),
    ("I’m not feeling good", "sad"),
    ("I’m feeling so anxious!", "stressed"),
    ("I feel like I have failed", "worthless"),
    ("I feel emotionless", "depressed"),
    ("Evening!", "evening"),
    ("Night!", "night"),
    ("Thank!", "thanks")
]
generalization_results = []
generalization_correct = 0
generalization_total = len(custom_sentences)
for sentence, expected_tag in custom_sentences:
    predicted_tag, confidence = predict_intent(sentence)
    correct = predicted_tag == expected_tag
    generalization_results.append((sentence, expected_tag, predicted_tag, confidence, correct))
    generalization_correct += correct
generalization_accuracy = (generalization_correct / generalization_total) * 100

# TEST 3: THRESHOLD TEST
threshold_results = []
thresh_value = 0.75  # Valore di soglia
threshold_correct = 0
threshold_total = len(custom_sentences)
for sentence, expected_tag in custom_sentences:
    predicted_tag, confidence = predict_intent(sentence)
    confidence_check = confidence >= thresh_value
    threshold_results.append((sentence, expected_tag, predicted_tag, confidence, confidence_check))
    threshold_correct += confidence_check
threshold_accuracy = (threshold_correct / threshold_total) * 100

# SALVATAGGIO DEI RISULTATI
output_path = "../test/testing_results.txt"
with open(output_path, "w", encoding='utf-8') as f:
    f.write("=== BASELINE TEST RESULTS ===\n")
    for res in baseline_results:
        f.write(
            f"Input: {res[0]} | Expected: {res[1]} | Predicted: {res[2]} | Confidence: {res[3]:.4f} | Correct: {res[4]}\n")
    f.write(f"Baseline Accuracy: {baseline_accuracy:.2f}%\n")

    f.write("\n=== GENERALIZATION TEST RESULTS ===\n")
    for res in generalization_results:
        f.write(
            f"Input: {res[0]} | Expected: {res[1]} | Predicted: {res[2]} | Confidence: {res[3]:.4f} | Correct: {res[4]}\n")
    f.write(f"Generalization Accuracy: {generalization_accuracy:.2f}%\n")

    f.write("\n=== THRESHOLD TEST RESULTS ===\n")
    for res in threshold_results:
        f.write(
            f"Input: {res[0]} | Expected: {res[1]} | Predicted: {res[2]} | Confidence: {res[3]:.4f} | Above Threshold: {res[4]}\n")
    f.write(f"Threshold Test Accuracy: {threshold_accuracy:.2f}%\n")

print("Testing completato. I risultati sono stati salvati in", output_path)