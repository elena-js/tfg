# PREPARACION DE DATOS ------------------------------------------------------------------------------

# librerias
import json
import re
import emoji

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from textblob import TextBlob

# función extraer emojis
def extract_emojis(str):
    return ''.join(c for c in str if c in emoji.UNICODE_EMOJI['en'])

# obtencion del array de comentarios a partir del json
f = open("comments.json", "r")
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
mentions = [] # array menciones
emojis = [] # array emojis

for comment in comments_in:
    
    # quitar emojis
    em = []
    e = extract_emojis(comment)
    for s in e:
        #if s not in em:
        em.append(s)
    comment = ''.join( x for x in comment if x not in em)
    # hacer lista de tokens
    tokens = word_tokenize(comment)
    # quitar menciones
    mention = []
    for i in range(len(tokens)):
        if tokens[i] == '@' and i < (len(tokens) - 1):
            mention.append(tokens[i+1])
            tokens[i] = ''
            tokens[i+1] = ''
        # limpiar texto
        TEXT_CLEANING_RE = "[^a-zA-Z0-9]+"
        tokens[i] = re.sub(TEXT_CLEANING_RE, '', tokens[i].lower())
    
    # quitar stopwords
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops]
    
    # stemming/lemmatization
    stemmer = PorterStemmer()
    bow = []
    for t in tokens:
        bow.append(stemmer.stem(t))
    
    # juntar palabras y volver a tokenizar para quitar espacios
    comment = (' '.join(bow))
    bow = word_tokenize(comment)
    comment = (' '.join(bow))
    # añadir variables a los arrays
    mentions.append(mention)
    wordlist.append(bow)
    cl_comments.append(comment)
    emojis.append(em)

# conTextBlob - polaridad (-1,1) - subjetividad(0,1)
sent = []
for comment in cl_comments:
    #print('\n'+comment)
    analysis = TextBlob(comment)
    analysis = analysis.sentiment
    #print(analysis)
    popularity = analysis.polarity
    #sent.append(popularity)
    if popularity >= 0.5:
        sent.append("pos")
    elif popularity <= -0.5:
        sent.append("neg")
    else:
        sent.append("neu")

# HACER CSV
import pandas as pd	

dict = {'comment': comments_in, 'cl_comment': cl_comments, 'wordlist': wordlist, 'emojis': emojis, 'sent': sent, 'user': users, 'mentioned': mentions} 
df = pd.DataFrame(dict) 
df.to_csv('data2.csv')
print('\ndataset creado correctamente')

#------------------------------------------------------------------------------------------------------

print()