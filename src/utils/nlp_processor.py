#Modulo per analisi del linguaggio naturale (tokenizzazione, intent detection).
import re

def process_input(user_input):
    """
    Pulisce e analizza l'input dell'utente.
    """
    # Pulisce l'input (rimuove caratteri speciali)
    cleaned_input = re.sub(r"[^\w\s]", "", user_input.lower())

    # Tokenizza l'input
    tokens = cleaned_input.split()

    # Rileva intenti (esempio semplice)
    if any(word in tokens for word in ["ciao", "salve", "hey"]):
        intent = "greeting"
    elif any(word in tokens for word in ["arrivederci", "addio"]):
        intent = "farewell"
    else:
        intent = "unknown"

    return {"tokens": tokens, "intent": intent}
