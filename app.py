from flask import Flask, render_template, request, jsonify
from chat import get_response
from googletrans import Translator

app = Flask(__name__)
app.secret_key = "your_secret_key"


translator = Translator()

def translate_to_english(text):
    detected_lang = translator.detect(text).lang  
    if detected_lang != 'en':
        translated_text = translator.translate(text, src=detected_lang, dest='en').text
        return translated_text, detected_lang
    return text, detected_lang

def translate_back(text, target_lang):
    if target_lang != 'en':
        translated_text = translator.translate(text, src='en', dest=target_lang).text
        return translated_text
    return text


@app.route('/')
def home():
    return render_template('index.html')

# Route to handle chat input
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    translated_input, detected_lang = translate_to_english(user_input)
    bot_response = get_response(translated_input)
    translated_bot_response = translate_back(bot_response, detected_lang)
    return jsonify({'answer': translated_bot_response})

if __name__ == "__main__":
    app.run(debug=True)
