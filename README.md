# Chatbot example

Chat over documents implementation based on [fully-local-pdf-chatbot](https://github.com/jacoblee93/fully-local-pdf-chatbot), this repo is not intended for production 

# Quick start

Install the dependencies and start the express server, make sure you have `ollama` running and downloaded [mistral](https://ollama.com/library/mistral)

```
yarn install
node app/index.js
```

Do a request to chat:

```
curl --location 'http://127.0.0.1:3000/chat' \
--header 'Content-Type: application/json' \
--data '{"message": "quien me puede dar llaves nuevas en caso de que las pierda?"}'

```
Example response:
```
{
    "success": true,
    "answer": " You can obtain new keys or remote controls from an authorized distributor."
}
```
