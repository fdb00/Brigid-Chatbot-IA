#Modulo per le risposte
def generate_response(processed_input):
    """
    Genera una risposta in base all'intento rilevato.
    """
    intent = processed_input.get("intent")

    if intent == "greeting":
        return "Ciao! Come posso aiutarti oggi?"
    elif intent == "farewell":
        return "A presto! Buona giornata!"
    else:
        return "Non sono sicuro di aver capito. Puoi riformulare?"
