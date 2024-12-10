#Contiene l'interfaccia principale del chatbot
from nlp_processor import process_input
from response_generator import generate_response

def main():
    print("Benvenuto nel chatbot! Scrivi 'esci' per terminare.")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "esci":
            print("Chatbot: Arrivederci!")
            break

        # Analizza l'input
        processed_input = process_input(user_input)

        # Genera la risposta
        response = generate_response(processed_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
