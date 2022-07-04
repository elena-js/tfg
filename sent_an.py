# ANÁLISIS DE SENTIMIENTOS --------------------------------------------------------------------------

# librerias
from nltk.sentiment.util import *
import pandas as pd	
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import joblib

# obtencion del dataset
data = pd.read_csv("data.csv") # ---------------------- CAMBIAR DATASET

# separar datos en train/test
x_train0, x_test0, y_train, y_test = train_test_split(data, data['sent'], test_size=0.2)

# para guardar en csv
x_train0.to_csv('train.csv') 
x_test0.to_csv('test.csv')

num_comments = x_test0.shape[0] # número de comentarios del test

# VECTORIZACIÓN - COUNTVECTORIZER / TFIDFVECTORIZER
vectorizer1 = CountVectorizer() # ngram_range=(1,2)
vectorizer2 = TfidfVectorizer()
#vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

# metiendo el comentario entero
train_vectors = vectorizer1.fit_transform(x_train0['cl_comment'].values.astype('U')).toarray()
test_vectors = vectorizer1.transform(x_test0['cl_comment'].values.astype('U')).toarray()

print(vectorizer1.get_feature_names_out())

# PARA METER AL SISTEMA X SERÁN COMENTARIOS (VECTORS), Y SERÁN LOS VALORES DE SENTIMIENTO

x_train = train_vectors
x_test = test_vectors

# CLASIFICADOR SVM ----------------------------------------------------------------------------------

# DEFINIR ALGORITMO ------------------------------------------------------------

# SVC
from sklearn.svm import SVC
algoritmo = SVC(kernel='linear')

# Logistic Regression
from sklearn.linear_model import LogisticRegression
#algoritmo = LogisticRegression()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
#algoritmo = DecisionTreeClassifier()

# Naive Bayes - NO
from sklearn.naive_bayes import GaussianNB
#algoritmo = GaussianNB()

# kNN
from sklearn.neighbors import KNeighborsClassifier
#algoritmo = KNeighborsClassifier(n_neighbors=5)

# RandomForest
from sklearn.ensemble import RandomForestClassifier
#algoritmo = RandomForestClassifier()

# ------------------------------------------------------------------------------

# cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
#cv = KFold(n_splits=3, random_state=1, shuffle=True)
#scores = cross_val_score(algoritmo, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# ENTRENAR MODELO
algoritmo.fit(x_train, y_train)
print('\nModelo entrenado')

# cargar modelo - si ya se ha entrenado
#modelo = joblib.load('svm_entrenado.pkl') 
#print('\nModelo cargado\n')

# REALIZAR PREDICCIÓN
y_pred = algoritmo.predict(x_test)

# OBTENCIÓN DE MÉTRICAS --------------------------------------------------------

# matriz confusión [v- f~ f+, (columnas son valores reales, filas son valores predecidos)
#                   f- v~ f+, 
#                   f- f~ v+] 
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('\nMatriz de confusión:') 
print(matriz)

# obtener report
# - de cada clase - precision, recall, f1-score, support (nº valores en realidad)
# - en total - accuracy, macro avg (mismo de antes), weighted avg (mismo de antes)
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print('\nReport:')
print(report)
#report2 = classification_report(y_test, y_pred, output_dict=True)
#print(report2['accuracy'])

# curva roc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
#fper, tper, thresholds = roc_curve(y_test, y_pred)
#plot_roc_curve(fper, tper)

# ANÁLISIS DATOS ---------------------------------------------------------------

# obtener + datos
neg_users = []
likes = []
mentioned = []
i = 0
while i < len(y_pred):
    if y_pred[i]== 'neg':
        c = x_test0.iloc[i]
        #print('\nComentario negativo:\n' + c['comment'])
        #if c['user'] not in neg_users:
            #neg_users.append([c['user'], 1])
        neg_users.append(c['user'])
        likes.append(c['likes'])
        mentioned.append(c['mentioned'])
        #print('likes:' + str(c['likes']))
        #else:
            #for user in neg_users:
                #if user[0] == c['user']:
                    #user[1] += 1          
    i+=1
#print('\nUsuarios que han realizado algún comentario negativo:')
#print(neg_users)

#print(mentioned)

# ver numero de comentarios negativos
num_neg = len(likes)
print('\nComentarios negativos: ' + str(num_neg))
p_neg = num_neg / num_comments
print('\n% comentarios negativos: ' + str(p_neg*100))

# usuarios que realizan + de 1 comentario negativo
users_1c = 0
users_2c = 0
users_3c = 0
users_5c = 0
for user in neg_users:
    if (neg_users.count(user) == 1):
        users_1c += 1
    if (neg_users.count(user) > 1):
        users_2c += 1
    if (neg_users.count(user) > 2):
        users_3c += 1
    if (neg_users.count(user) > 5):
        users_5c += 1
print('usuarios que realizan 1 comentario negativo:')
print(users_1c)
print('usuarios que realizan + de 1 comentario negativo:')
print(users_2c)
print('usuarios que realizan + de 2 comentarios negativos:')
print(users_3c)
print('usuarios que realizan + de 5 comentarios negativos:')
print(users_5c)

# suma de n_likes - usuarios que participan indirectamente
n_likes = sum(likes)
print('usuarios que participan indirectamente:')
print(n_likes)

# porcentaje de comentarios negativos con mención
n_mentions = 0
for mention in mentioned:
    if mention != ' ' and mention != 'kourtneykardash':
        n_mentions +=1
p_mentions = n_mentions / num_neg
print('porcentaje de comentarios negativos con mención:')
print(p_mentions)

# comentarios negativos dirigidos al perfil analizado (sin menciones)
com_neg_victima = num_neg - n_mentions
#print('comentarios negativos dirigidos al perfil analizado:')
#print(com_neg_victima)

# guardar modelo
#joblib.dump(algoritmo, 'svm_entrenado.pkl')
print('\nModelo guardado')

#------------------------------------------------------------------------------------------------------
print()