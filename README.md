# Brigid - Chatbot AI

Brigid è un chatbot basato su intelligenza artificiale progettato per interagire con gli utenti e rispondere a domande basandosi su intenti predefiniti. Il chatbot utilizza BERT per l'elaborazione del linguaggio naturale e segue il processo CRISP-DM per la gestione del ciclo di sviluppo.

## 📌 Funzionalità Principali

- Comprensione ed elaborazione del linguaggio naturale con BERT.

- Dataset esteso con 223 intenti e 50 pattern per tag, generati anche con data augmentation.

- Gestione della conversazione con modello addestrato o API Gemini.

- Testing con Baseline Test, Generalization Test e Threshold Test.

## 🚀 Setup e Avvio del Progetto su PyCharm

### 1️⃣ Clonare il Repository

Aprire il terminale e clonare il progetto GitHub:

" git clone https://github.com/tuo-utente/brigid-chatbot.git 
cd brigid-chatbot "

### 2️⃣ Creare un Ambiente Virtuale

Su Windows:

- " python -m venv .venv "
- " .venv\Scripts\activate "

Su macOS/Linux:

- " python3 -m venv .venv "
- " source .venv/bin/activate "

### 3️⃣ Installare le Dipendenze

Dopo aver attivato l'ambiente virtuale, installare i pacchetti richiesti:

- " pip install -r requirements.txt "

#### Importante : Generate la vostra API Key per Gemini da https://aistudio.google.com/

Copiate la chiave e scrivete sul terminale
- " setx SECRET_API_KEY "INSERIRE QUI LA API KEY" (Windows) "
- " export SECRET_API_KEY="INSERIRE QUI LA API KEY" (Unix/Unix-like) "

Riavvia PyCharm e ora dovrebbe funzionare!

### 4️⃣ Addestrare il Modello (non necessario)

Per addestrare il modello NLP, eseguire:

- " python src/train/pretrain.py "

Il modello addestrato verrà salvato nella cartella models/.

#### Attenzione: il modello è già presente nella cartella src/models, prima di eseguire pretrain cancellare il vecchio modello chatbot_model.keras con il file classes.pkl

### 5️⃣ Avviare il Chatbot

Dopo l'addestramento, è possibile eseguire il chatbot nel terminale:

" python src/utils/chatbot_terminal.py "

Il chatbot sarà pronto per rispondere ai messaggi dell'utente!

### 📊 Testing del Modello

Il chatbot viene testato con tre strategie:

- Baseline Test: verifica l'accuratezza sugli intenti appresi.

- Generalization Test: verifica la capacità del modello di riconoscere frasi nuove.

- Threshold Test: analizza la confidenza della predizione.

Per eseguire il testing:

- " python test/testing_chatbot.py "

I risultati saranno salvati in test/testing_results.txt.

## 📈 Possibili Miglioramenti

- Espansione del dataset per migliorare la generalizzazione.

- Fine-tuning avanzato su un dataset più ampio.

- Supporto multilingua per interazioni in più lingue.

- Deploy su cloud per integrazione con applicazioni web.

