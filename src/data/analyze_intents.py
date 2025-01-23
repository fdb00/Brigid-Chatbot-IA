import json

def analyze_intents(json_file_path):
    try:
        # Carica il file JSON
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Verifica che ci sia la chiave "intents"
        if "intents" not in data:
            print("Errore: il file JSON non contiene la chiave 'intents'.")
            return

        # Conta il numero totale di tag
        total_tags = len(data["intents"])
        print(f"Numero totale di tag: {total_tags}")
        print("-" * 50)

        # Estrai la lista dei tag
        tag_list = [intent.get("tag", "Tag non specificato") for intent in data["intents"]]
        print("Lista dei tag:")
        print(tag_list)
        print("-" * 50)

        # Analizza i tag
        print("Analisi dei tag nel database:")
        print("-" * 50)

        for intent in data["intents"]:
            tag = intent.get("tag", "Tag non specificato")
            patterns = intent.get("patterns", [])
            responses = intent.get("responses", [])

            print(f"Tag: {tag}")
            print(f"  Numero di patterns: {len(patterns)}")
            print(f"  Numero di responses: {len(responses)}")
            print("-" * 50)

    except FileNotFoundError:
        print(f"Errore: il file '{json_file_path}' non è stato trovato.")
    except json.JSONDecodeError:
        print(f"Errore: il file '{json_file_path}' non è un JSON valido.")

if __name__ == "__main__":
    # Sostituisci il percorso con il file JSON contenente il tuo database
    json_file_path = "./intents.json"

    analyze_intents(json_file_path)
