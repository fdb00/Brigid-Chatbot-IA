# Brigid - Chatbot AI

Brigid è una chatbot basata sull'IA progettata per interagire con gli utenti e rispondere a domande basandosi su intenti predefiniti. La chatbot utilizza BERT per l'elaborazione del linguaggio naturale e segue il processo CRISP-DM per la gestione del ciclo di sviluppo. Inoltre, è capace di rispondere alla maggior parte degli input grazie all'integrazione del LLM Gemini, che risponderà nel caso in cui nel modello non ci siano intenti adatti.

Progettata in ![Python 3.11.1](https://img.shields.io/badge/Python-3.11.1-blue?logo=python)

## 📌 Funzionalità Principali

- Comprensione ed elaborazione del linguaggio naturale con BERT.

- Dataset esteso con 223 intenti e 50 pattern per tag, generati anche con data augmentation.

- Gestione della conversazione con modello addestrato o API Gemini.

- Testing con Baseline Test, Generalization Test e Threshold Test.

## 🚀 Setup e Avvio del Progetto su PyCharm

### 1️⃣ Clonare il Repository

Aprire il terminale e clonare il progetto GitHub:

```
git clone https://github.com/fdb00/Brigid-Chatbot-IA.git
```

### 2️⃣ Aprire la cartella di Brigid clonata in PyCharm

### 3️⃣ Installare le Dipendenze

Dopo aver attivato l'ambiente virtuale, installare i pacchetti richiesti:

```
pip install -r requirements.txt
```

#### Importante: Generate la vostra API Key per Gemini da https://aistudio.google.com/

Copiate la chiave e scrivete sul terminale
```
setx SECRET_API_KEY "INSERIRE QUI LA API KEY" \\ Windows
export SECRET_API_KEY="INSERIRE QUI LA API KEY" \\ Unix/Unix-like
```

Riavvia PyCharm e ora dovrebbe funzionare!

### 4️⃣ Addestrare il Modello (non necessario)

Per addestrare il modello NLP, eseguire il file:
```
src/train/pretrain.py
```

Il modello addestrato verrà salvato nella cartella models/.

#### Attenzione: il modello è già presente nella cartella src/models, prima di eseguire pretrain cancellare il vecchio modello chatbot_model.keras con il file classes.pkl

### 5️⃣ Avviare il Chatbot

Dopo l'addestramento, è possibile eseguire il chatbot nel terminale cliccando il tasto "Run" in altro a destra in PyCharm.

La chatbot sarà pronta per rispondere ai messaggi dell'utente!

### 📊 Testing del Modello

Il chatbot viene testato con tre strategie:

- Baseline Test: verifica l'accuratezza sugli intenti appresi.

- Generalization Test: verifica la capacità del modello di riconoscere frasi nuove.

- Threshold Test: analizza la confidenza della predizione.

Per eseguire il testing, runnare il file:

```
test/testing_chatbot.py
```

I risultati saranno salvati in test/testing_results.txt.

## 📈 Possibili Miglioramenti

- Espansione del dataset per migliorare la generalizzazione.

- Fine-tuning avanzato su un dataset più ampio.

- Supporto multilingua per interazioni in più lingue.

- Deploy su cloud per integrazione con applicazioni web.

