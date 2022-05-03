# PREPARACION DE DATOS ------------------------------------------------------------------------------

# librerias
import json
import re
import string

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

# obtencion del array de comentarios a partir del json
f = open("b.json", "r")
content = f.read()
info_com = json.loads(content)

# obtencion texto comentarios y usuarios
comments_in = []
comments = []
users = []
for item in info_com:
    comments.append(item['text'])
    users.append(item['owner']['username'])
    if len(item['answers']) != 0:
        for i in item['answers']:
            comments.append(i['text'])
            users.append(i['owner']['username'])
comments_in.extend(comments)

# quitar menciones (tambien se quitan emojis)
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
for i in range(len(comments)):
    comments[i] = re.sub(TEXT_CLEANING_RE, ' ', str(comments[i]).lower()).strip()

# limpiar texto (dejar emojis)
'''
TEXT_CLEANING_RE = "@\S+[^A-Za-z0-9]+"
for i in range(len(comments)):
    comments[i] = re.sub(TEXT_CLEANING_RE, ' ', str(comments[i]).lower()).strip().translate(str.maketrans('','', string.punctuation)).lower()
print(comments)
'''
# analisis de sentimientos (basico)
s_an = SentimentIntensityAnalyzer()
sent = []
for comment in comments:
    #print('\n'+comment)
    scores = s_an.polarity_scores(comment)
    #for key in scores:
        #print(key, ': ', scores[key])
    #sent.append(scores['compound'])
    if scores['compound'] >= 0.5:
        sent.append("pos")
    elif scores['compound'] <= -0.5:
        sent.append("neg")
    else:
        sent.append("neu")

# HACER CSV
import pandas as pd	

dict = {'comment': comments_in, 'cl_comment': comments, 'user': users, 'sent': sent} 
df = pd.DataFrame(dict) 
df.to_csv('data.csv')
print('\ndataset creado correctamente')

#------------------------------------------------------------------------------------------------------

print()

# expresiones interesantes
'''
words_cleaned = [word for word in words_filtered
    if 'http' not in word
    and not word.startswith('@')
    and not word.startswith('#')
    and word != 'RT']
'''