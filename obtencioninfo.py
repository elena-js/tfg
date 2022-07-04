# librerias
from nltk.sentiment.util import *
import pandas as pd	
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# PUBLICACIÓN 1 ----------------------------------------------------------------------------------
# obtencion del dataset 1
data1 = pd.read_csv("car/data_personal.csv") # ------------- CAMBIAR PUBLICACIÓN 1
# separar datos en train/test
x_train01, x_test01, y_train1, y_test1 = train_test_split(data1, data1['sent'], test_size=0.2)
# para guardar en csv
x_train01.to_csv('train1.csv') 
x_test01.to_csv('test1.csv')
# número de comentarios del test
#num_comments1 = x_test01.shape[0] 
# VECTORIZACIÓN - COUNTVECTORIZER
vectorizer = CountVectorizer()
x_train1 = vectorizer.fit_transform(x_train01['cl_comment'].values.astype('U')).toarray()
x_test1 = vectorizer.transform(x_test01['cl_comment'].values.astype('U')).toarray()
# DEFINIR ALGORITMO 
from sklearn.svm import SVC
algoritmo = SVC(kernel='linear')
# ENTRENAR MODELO
algoritmo.fit(x_train1, y_train1)
# REALIZAR PREDICCIÓN
y_pred1 = algoritmo.predict(x_test1)
print('\nPublicación 1 analizada')

# PUBLICACIÓN 2 ----------------------------------------------------------------------------------
# obtencion del dataset 2
data2 = pd.read_csv("car/data_puma.csv") # ------------- CAMBIAR PUBLICACIÓN 2
# separar datos en train/test
x_train02, x_test02, y_train2, y_test2 = train_test_split(data2, data2['sent'], test_size=0.2)
# para guardar en csv
x_train02.to_csv('train2.csv') 
x_test02.to_csv('test2.csv')
# número de comentarios del test
#num_comments2 = x_test02.shape[0] 
# VECTORIZACIÓN - COUNTVECTORIZER
vectorizer = CountVectorizer()
x_train2 = vectorizer.fit_transform(x_train02['cl_comment'].values.astype('U')).toarray()
x_test2 = vectorizer.transform(x_test02['cl_comment'].values.astype('U')).toarray()
# DEFINIR ALGORITMO 
from sklearn.svm import SVC
algoritmo = SVC(kernel='linear')
# ENTRENAR MODELO
algoritmo.fit(x_train2, y_train2)
# REALIZAR PREDICCIÓN
y_pred2 = algoritmo.predict(x_test2)
print('\nPublicación 2 analizada')

# ANÁLISIS DATOS -----------------------------------------------------------------------------------------
from decimal import *
# PUBLICACIÓN 1 -------------------------------------------------------

# USUARIOS AUTORES DE LOS COMENTARIOS NEGATIVOS
neg_users1 = []

# USUARIOS MENCIONADOS
mentioned1 = []

likes1 = []

i = 0
while i < len(y_pred1):
    if y_pred1[i]== 'neg':
        c = x_test01.iloc[i]
        neg_users1.append(c['user'])
        likes1.append(c['likes'])
        mentioned1.append(c['mentioned'])  
    i+=1

# DATOS DE PRUEBA - len el nº de com neg
'''
neg_users1 = ['user1', 'user2', 'user3', 'user1', 'user4']
mentioned1 = ['user5', 'user6', 'user7', ' ', 'user8']
likes1 = [0, 3, 0, 5, 20]
'''
# NÚMERO DE COMENTARIOS NEGATIVOS
com_neg1 = len(likes1)

# NÚMERO DE LIKES DE LOS COMENTARIOS NEGATIVOS
n_likes1 = sum(likes1)

# PUBLICACIÓN 2 -------------------------------------------------------

# USUARIOS AUTORES DE LOS COMENTARIOS NEGATIVOS
neg_users2 = []

# USUARIOS MENCIONADOS
mentioned2 = []

likes2 = []

i = 0
while i < len(y_pred2):
    if y_pred2[i]== 'neg':
        c = x_test02.iloc[i] 
        neg_users2.append(c['user'])
        likes2.append(c['likes'])
        mentioned2.append(c['mentioned'])        
    i+=1

# DATOS DE PRUEBA - len el nº de com neg
'''
neg_users2 = ['user11', 'user2', 'user13', 'user1', 'user4', ' ', 'user1']
mentioned2 = ['user15', 'user6', 'user7', ' ', 'user18', 'user19',' ']
likes2 = [0, 3, 0, 5, 20, 0, 2]
'''
# NÚMERO DE COMENTARIOS NEGATIVOS
com_neg2 = len(likes2)

# NÚMERO DE LIKES DE LOS COMENTARIOS NEGATIVOS
n_likes2 = sum(likes2)

# CÁLCULO DE LA INFORMACIÓN RELEVANTE ---------------------------------------------------------------------------

# DIFERENCIA % DE COMENTARIOS NEGATIVOS -----------------------------------------------------------
com_neg = com_neg2 / com_neg1 * 100 - 100
print(com_neg1)
print(com_neg2)
print()
if (com_neg > 0): 
    var = 'más'
else:
    var = 'menos'
    com_neg *= -1
print('\nEn la publicación 2 hay un ' + str(com_neg) + '% ' + var + ' de comentarios negativos que en la publicación 1')

# TOP USUARIOS QUE REALIZAN + COMENTARIOS NEGATIVOS ------------------------------------------------
neg_users = neg_users1 + neg_users2
users_1c = []
users_2c = []
users_5c = []
for user in neg_users:
    if (neg_users.count(user) > 1):
        if (user not in users_1c):
            users_1c.append(user)
    if (neg_users.count(user) > 2):
        if (user not in users_2c):
            users_2c.append(user)
    if (neg_users.count(user) > 5):
        if (user not in users_5c):
            users_5c.append(user)
print('\nUsuarios que realizan + de 1 comentario negativo:')
print(users_1c)
print('\nUsuarios que realizan + de 2 comentarios negativos:')
print(users_2c)
print('\nUsuarios que realizan + de 5 comentarios negativos:')
print(users_5c)

# USUARIOS QUE PARTICIPAN DE MANERA INDIRECTA -------------------------------------------------
# publicación 1
print('\nEn la publicación 1:')
ind1 = n_likes1 / com_neg1
if (ind1 > 1): 
    print('Hay más ataques indirectos que directos en un factor x' + str(ind1))
else:
    ind1 = com_neg1 / n_likes1
    print('Hay más ataques directos que indirectos en un factor x' + str(ind1))

# publicación 2
print('\nEn la publicación 2:')
ind2 = n_likes2 / com_neg2
if (ind2 > 1): 
    print('Hay más ataques indirectos que directos en un factor x' + str(ind2))
else:
    ind2 = com_neg2 / n_likes2
    print('Hay más ataques directos que indirectos en un factor x' + str(ind2))

# PORCENTAJE DE COMENTARIOS NEGATIVOS CON MENCIÓN -----------------------------------------------
# publicacion1
print('\nEn la publicación 1:')
n_mentions1 = 0
for mention in mentioned1:
    if mention != ' ':
        n_mentions1 +=1
p_mentions1 = n_mentions1 / com_neg1 * 100
print('El ' + str(p_mentions1) + '% de los comentarios negativos contienen alguna mención a otro usuario')
# mención a otros usuarios
n_mentions_otro1 = 0
for mention in mentioned1:
    if mention != ' ' and mention != 'caradelevingne ': # CAMBIAR EL USUARIO
        n_mentions_otro1 +=1
p_mentions_otro1 = n_mentions_otro1 / n_mentions1 * 100
print('De estas menciones un ' + str(p_mentions_otro1) + '% son de usuarios externos')

# publicacion2
print('\nEn la publicación 2:')
n_mentions2 = 0
for mention in mentioned2:
    if mention != ' ':
        n_mentions2 +=1
p_mentions2 = n_mentions2 / com_neg2 * 100
print('El ' + str(p_mentions2) + '% de los comentarios negativos contienen alguna mención a otro usuario')
# mención a otros usuarios
n_mentions_otro2 = 0
for mention in mentioned2:
    if mention != ' ' and mention != 'caradelevingne ': # CAMBIAR EL USUARIO
        n_mentions_otro2 +=1
p_mentions_otro2 = n_mentions_otro2 / n_mentions2 * 100
print('De estas menciones un ' + str(p_mentions_otro2) + '% son de usuarios externos')

# TOP USUARIOS MÁS MENCIONADOS ------------------------------------------------------------------
mentioned = mentioned1 + mentioned2
mentioned_1c = []
mentioned_2c = []
mentioned_5c = []
for user in mentioned:
    if user != ' ':
        if (mentioned.count(user) > 1):
            if (user not in mentioned_1c):
                mentioned_1c.append(user)
        if (mentioned.count(user) > 2):
            if (user not in mentioned_2c):
                mentioned_2c.append(user)
        if (mentioned.count(user) > 5):
            if (user not in mentioned_5c):
                mentioned_5c.append(user)
print('\nUsuarios mencionados en + de 1 comentario negativo:')
print(mentioned_1c)
print('\nUsuarios mencionados en + de 2 comentarios negativos:')
print(mentioned_2c)
print('\nUsuarios mencionados en + de 5 comentarios negativos:')
print(mentioned_5c)

#------------------------------------------------------------------------------------------------------
print()