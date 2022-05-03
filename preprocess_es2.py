# librerias
import string
import nltk
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
import re

# funciones

stemmer = PorterStemmer()
spanish_stemmer = SnowballStemmer('spanish')
lemmatizer = WordNetLemmatizer()

# obtencion del array de comentarios a partir del json
f = open("c.json", "r")
content = f.read()
info_com = json.loads(content)

# obtencion texto comentarios y usuarios
comments_in = []
comments = []
users = []
for item in info_com:
    comments_in.append(item['text'])
    users.append(item['owner']['username'])
    if len(item['answers']) != 0:
        for i in item['answers']:
            comments_in.append(i['text'])
            users.append(i['owner']['username'])
comments.extend(comments_in)

TEXT_CLEANING_RE = "@\S+[^A-Za-z0-9]+"
for i in range(len(comments)):
    comments[i] = re.sub(TEXT_CLEANING_RE, ' ', str(comments[i]).lower()).strip().translate(str.maketrans('','', string.punctuation)).lower()

# variables del preprocess
wordlist = [] # array bow
freq = [] # array frecuencias de aparicion
cl_comments = [] # array comentarios limpios

for comment in comments:

    # quitar signos puntuacion y mayúsculas
    b = comment.translate(str.maketrans('','', string.punctuation)).lower()

    # hacer lista de tokens
    tokens = word_tokenize(b)

    # quitar stopwords
    bow = tokens[:]
    s = stopwords.words('spanish')
    for token in tokens:
        if token in s:
            bow.remove(token)

    wordlist.append(bow) # añadir al array de wordlist
    
    # calcular frecuencia de aparicion
    fd = nltk.FreqDist(bow)
    f = []
    for key,val in fd.items():
        f.append(str(val))
    freq.append(f) # añadir al array de frecuencias

    # unir palabras
    cc = (' '.join(bow))
    cl_comments.append(cc)

#-----------------------------------------------------------------------------------------------

# analisis de sentimientos (basico) EN ESPAÑOL
from sentiment_analysis_spanish import sentiment_analysis
sentiment = sentiment_analysis.SentimentAnalysisSpanish() # (0,1)

sent = []
for comment in cl_comments:
    sent.append(sentiment.sentiment(comment))

# HACER CSV
import pandas as pd	

dict = {'user': users, 'comment': comments_in, 'cl_comment': cl_comments, 'wordlist': wordlist, 'freq': freq, 'sent': sent} 
df = pd.DataFrame(dict) 
df.to_csv('data_es2.csv')
print('\ndataset creado correctamente')

#------------------------------------------------------------------------------------------------------

print()