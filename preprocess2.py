# PREPARACION DE DATOS ------------------------------------------------------------------------------

# librerias
import json
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

# obtencion del array de comentarios a partir del json
f = open("b.json", "r")
content = f.read()
info_com = json.loads(content)

# obtencion texto comentarios y usuarios
comments_in = []
users = []
for item in info_com:
    comments_in.append(item['text'])
    users.append(item['owner']['username'])
    if len(item['answers']) != 0:
        for i in item['answers']:
            comments_in.append(i['text'])
            users.append(i['owner']['username'])

# variables del preprocess
wordlist = [] # array bow
freq = [] # array frecuencias de aparicion -- TODAVÍA NO UTILIZADO
cl_comments = [] # array comentarios limpios

for comment in comments_in:
    # limpiar texto (dejar emojis)
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    comment = re.sub(TEXT_CLEANING_RE, ' ', str(comment).lower()).strip().translate(str.maketrans('','', string.punctuation)).lower()
    # hacer lista de tokens
    tokens = word_tokenize(comment)
    # quitar stopwords
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops]
    # stemming/lemmatization
    stemmer = PorterStemmer()
    bow = []
    for t in tokens:
        bow.append(stemmer.stem(t))
    # juntar palabras
    comment = (' '.join(bow))
    # añadir variables a los arrays
    wordlist.append(bow)
    cl_comments.append(comment)

# conTextBlob - polaridad (-1,1) - subjetividad(0,1)
from textblob import TextBlob
sent = []
for comment in cl_comments:
    #print('\n'+comment)
    analysis = TextBlob(comment)
    analysis = analysis.sentiment
    #print(analysis)
    popularity = analysis.polarity
    sent.append(popularity)

# HACER CSV
import pandas as pd	

dict = {'comment': comments_in, 'cl_comment': cl_comments, 'user': users, 'sent': sent} 
df = pd.DataFrame(dict) 
df.to_csv('data2.csv')
print('\ndataset creado correctamente')

#------------------------------------------------------------------------------------------------------

print()