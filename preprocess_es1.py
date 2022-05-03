# PREPARACION DE DATOS ------------------------------------------------------------------------------

# librerias
import string
import nltk
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
import re

# obtencion del array de comentarios a partir del json
f = open("c.json", "r")
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
    stops = set(stopwords.words('spanish'))
    tokens = [t for t in tokens if t not in stops]
    # juntar palabras
    comment = (' '.join(tokens))
    # añadir variables a los arrays
    wordlist.append(tokens)
    cl_comments.append(comment)

# analisis de sentimientos (sas)
from sentiment_analysis_spanish import sentiment_analysis
sentiment = sentiment_analysis.SentimentAnalysisSpanish() # (0,1)

sent = []
for comment in cl_comments:
    sent.append(sentiment.sentiment(comment))

# HACER CSV
import pandas as pd	

dict = {'comment': comments_in, 'cl_comment': cl_comments, 'user': users, 'sent': sent} 
df = pd.DataFrame(dict) 
df.to_csv('data_es1.csv')
print('\ndataset creado correctamente')

#------------------------------------------------------------------------------------------------------

print()