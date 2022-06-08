# PREPARACION DE DATOS ------------------------------------------------------------------------------

# librerias
import json
import re
import emoji
#import spacy

import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

# obtencion del array de comentarios a partir del json
f = open("comments.json", "r")
content = f.read()
info_com = json.loads(content)

# obtencion texto comentarios, usuarios, likes
comments_in = []
users = []
likes = []
for item in info_com:
    comments_in.append(item['text'])
    users.append(item['owner']['username'])
    likes.append(item['likes_count'])
    if len(item['answers']) != 0:
        for i in item['answers']:
            comments_in.append(i['text'])
            users.append(i['owner']['username'])
            likes.append(item['likes_count'])

# variables del preprocess
wordlist = [] # array bow
freq = [] # array frecuencias de aparicion -- TODAVÍA NO UTILIZADO
cl_comments = [] # array comentarios limpios
mentions = [] # array menciones
emojis = [] # array emojis
n_words = [] # array número de palabras
n_emojis = [] # array número de emojis

for comment in comments_in:
    
    # quitar emojis
    em = ''
    em_array = []
    comment_em = ''.join(c for c in comment if c in emoji.UNICODE_EMOJI['en'])
    for e in comment_em:
        #if s not in em:
        em_array.append(e)
        em += e
    if not em:
        em = ' '
    comment = ''.join( x for x in comment if x not in em_array)

    # quitar URLs
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    comment = re.sub(urlPattern,' ',comment)

    # hacer lista de tokens
    tokens = word_tokenize(comment)

    #mention = []
    mention = ''
    for i in range(len(tokens)):

        # quitar menciones
        if tokens[i] == '@' and i < (len(tokens) - 1):
            #mention.append(tokens[i+1])
            mention = mention + tokens[i+1] + ' '
            tokens[i] = ''
            tokens[i+1] = ''
        
        # quedarse con negación
        if tokens[i] == 'n\'t':
            tokens[i] = 'not'
        if tokens[i] == 'haven' or tokens[i] == 'didn' or tokens[i] == 'wouldn' or tokens[i] == 'mightn' or tokens[i] == 'mustn' or tokens[i] == 'weren' or tokens[i] == 'hadn' or tokens[i] == 'shouldn' or tokens[i] == 'isn' or tokens[i] == 'wasn' or tokens[i] == 'doesn' or tokens[i] == 'couldn' or tokens[i] == 'hasn' or tokens[i] == 'shan' or tokens[i] == 'needn' or tokens[i] == 'aren' or tokens[i] == 'won' or tokens[i] == 'don' :
            tokens[i] = 'not'

        # limpiar texto
        TEXT_CLEANING_RE = "[^a-zA-Z]+"
        #TEXT_CLEANING_RE = "[^a-zA-Z0-9]+"  -- con números
        tokens[i] = re.sub(TEXT_CLEANING_RE, '', tokens[i].lower())
    
    # quitar stopwords
    stops = set(stopwords.words('english'))
    stops.remove('not')
    stops.remove('no')

    tokens = [t for t in tokens if t not in stops]
    
    # stemming/lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    bow = []
    '''
    for t in tokens:
        bow.append(stemmer.stem(t))
        #bow.append(lemmatizer.lemmatize(t))
    '''
    # juntar palabras y volver a tokenizar para quitar espacios
    #comment = (' '.join(bow))
    comment = (' '.join(tokens))
    bow = word_tokenize(comment)
    n_words.append(len(bow))
    comment = (' '.join(bow))
    if not comment:
        comment = ' '
    if not mention:
        mention = ' '
    # añadir variables a los arrays
    mentions.append(mention)
    wordlist.append(bow)
    cl_comments.append(comment)
    emojis.append(em)

# ANÁLISIS DE SENTIMIENTO DE EMOJIS
sent_emojis = pd.read_csv("sent_emojis.csv")
sent_emojis = [sent_emojis['Emoji'].values.tolist(), sent_emojis['sent'].values.tolist()]
# sent_emojis[0->e 1->s][pos e])

# asignar valor de sentimiento a cada grupo de emojis
sent_em = []
for a in range(len(emojis)):
    s = []
    for e in emojis[a]:
        for i in range(len(sent_emojis[0])):
            if e == sent_emojis[0][i]:
                s.append(sent_emojis[1][i])
    total = 0
    if s:
        total = sum(s) / len(s)
    sent_em.append(total)
    n_emojis.append(len(s))

# ANÁLISIS DE SENTIMIENTO (NLTK)
s_an = SentimentIntensityAnalyzer()
sent = []
sent_num = []
for c in range(len(cl_comments)):
    scores = s_an.polarity_scores(cl_comments[c])
    # sin tener en cuenta emojis:
    if scores['compound'] >= 0.4:
        sent.append("pos")
    elif scores['compound'] <= -0.4:
        sent.append("neg")
    else:
        sent.append("neu")
    
    if (n_emojis[c] + n_words[c]) != 0:
        fw = n_words[c] / (n_emojis[c] + n_words[c])
        fe = n_emojis[c] / (n_emojis[c] + n_words[c])
        s = fw * scores['compound'] + fe * sent_em[c]
    else:
        s = 0
    sent_num.append(s)
    
    # teniendo en cuenta emojis:
    '''
    if s >= 0.5:
        sent.append("pos")
    elif s <= -0.5:
        sent.append("neg")
    else:
        sent.append("neu")
    '''

# HACER CSV
import pandas as pd

dict = {'comment': comments_in, 'cl_comment': cl_comments, 'wordlist': wordlist, 'emojis': emojis, 'sent_num': sent_num, 'sent': sent, 'user': users, 'mentioned': mentions, 'likes': likes} 
df = pd.DataFrame(dict) 
df.to_csv('data.csv')
#df.to_csv('data.csv', sep=';')
print('\ndataset creado correctamente')

#------------------------------------------------------------------------------------------------------

print()