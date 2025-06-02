import requests

url = "http://localhost:8000/predict"

while True:
    text = input("Inserisci una frase (o 'exit' per uscire): ")
    if text.lower() == "exit":
        break

    response = requests.post(url, json={"text": text})
    if response.status_code == 200:
        print("Emozioni rilevate:", response.json()["emotions"])
    else:
        print("Errore:", response.text)
