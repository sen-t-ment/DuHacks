import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import os
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import nltk
from nltk.corpus import stopwords
import numpy as np
from keras.models import load_model

nltk.download('punkt')

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "DatasetF.csv"),delimiter=',',encoding='latin-1')
#print(df.head())
df.index = range(4845)
df['Message'].apply(lambda x: len(x.split(' '))).sum()
sentiment  = {'positive': 0,'neutral': 1,'negative':2} 
df.sentiment = [sentiment[item] for item in df.sentiment] 

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['Message'] = df['Message'].apply(cleanText)

train, test = train_test_split(df, test_size=0.000001 , random_state=42)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            #if len(word) < 0:
            if len(word) <= 0:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)

# The maximum number of words to be used. (most frequent)
max_fatures = 500000

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50

tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Message'].values)
X = tokenizer.texts_to_sequences(df['Message'].values)
X = pad_sequences(X)

d2v_model = Doc2Vec(dm=1, dm_mean=1,vector_size=20,  window=8, min_count=1, workers=1, alpha=0.065, min_alpha=0.065)
d2v_model.build_vocab([x for x in tqdm(train_tagged.values)])

embedding_matrix = np.zeros((len(d2v_model.wv.key_to_index)+ 1, 20))

for i, vec in enumerate(d2v_model.dv.vectors):
    while i in vec <= 1000:
    #print(i)
    #print(model.docvecs)
          embedding_matrix[i]=vec
    #print(vec)
    #print(vec[i])


# init layer
model = Sequential()

# emmbed word vectors
model.add(Embedding(len(d2v_model.wv.key_to_index)+1,20,input_length=X.shape[1],weights=[embedding_matrix],trainable=True))

# learn the correlations
def split_input(sequence):
     return sequence[:-1], tf.reshape(sequence[1:], (-1,1))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(3,activation="softmax"))

# output model skeleton
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])

Y = pd.get_dummies(df['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
# print(X_train.shape,Y_train.shape)
# print(X_test.shape,Y_test.shape)
batch_size = 32
history=model.fit(X_train, Y_train, epochs =50, batch_size=batch_size, verbose = 2)

_, train_acc = model.evaluate(X_train, Y_train, verbose=2)
_, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))


rounded_labels=np.argmax(Y_test, axis=1)

validation_size = 610

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)

print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

message = [""]
def get_value(str_var):
    message = [f"{str_var}"]
    seq = tokenizer.texts_to_sequences(message)

    padded = pad_sequences(seq, maxlen=X.shape[1], dtype='int32', value=0)

    pred = model.predict(padded)

    labels = ['0','1','2']
    l=labels[np.argmax(pred)]


    if l=='0':
        return f"""Statement is Positive.\nTraining Accuracy: {int(train_acc * 100)}%\nand\nTesting Accuracy: {int(test_acc * 100)}%"""
        print("POSITIVE")
    elif l == '1':
        return f"""Statement is Neutral.\nTraining Accuracy: {int(train_acc * 100)}%\nand\nTesting Accuracy: {int(test_acc * 100)}%"""
        print("NEUTRAL")
    else:
        return f"""Statement is Negative.\nTraining Accuracy: {int(train_acc * 100)}%\nand\nTesting Accuracy: {int(test_acc * 100)}%"""
        print("NEGATIVE")

model.save("MyModel.h5")

# fetch financial news keywords from twitter and compare it with user's input by first tokenizing the user's input and the comparing it with the fetched keyword

