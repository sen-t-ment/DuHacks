import main
from flask import Flask, render_template, redirect, url_for, request,jsonify
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
app = Flask(__name__)
app_name ="Finance Sentiment Analysis"
@app.route("/")
def hello_world():
    return render_template('index.html',app_name=app_name)

@app.route('/_predict_search')
def pred():
    query = request.args.get('a', 0, type=str)
    Answer = main.get_value(str_var=query).split('\n') 
    print(Answer)   
    return jsonify(result=Answer)


if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        print("Error Occured!")
    
