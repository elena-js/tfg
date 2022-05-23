# ANÁLISIS DE SENTIMIENTOS --------------------------------------------------------------------------

# librerias
from nltk.sentiment.util import *
import pandas as pd	
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# obtencion del dataset
data = pd.read_csv("data.csv")

# separar datos en train/test (justificar %)
x_train0, x_test0, y_train, y_test = train_test_split(data, data['sent'], test_size=0.2)

# para guardar en csv
x_train0.to_csv('train.csv') # ------------------------ ES NECESARIO?
x_test0.to_csv('test.csv')

# VECTORIZACIÓN - COUNTVECTORIZER / TFIDFVECTORIZER
vectorizer1 = CountVectorizer()
vectorizer2 = TfidfVectorizer()
#vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

train_vectors = vectorizer1.fit_transform(x_train0['wordlist'].values.astype('U'))
test_vectors = vectorizer1.transform(x_test0['wordlist'].values.astype('U'))

# PARA METER AL SISTEMA X SERÁN COMENTARIOS (VECTORS), Y SERÁN LOS VALORES DE SENTIMIENTO

x_train = train_vectors
x_test = test_vectors

# CLASIFICADOR SVM ----------------------------------------------------------------------------------

# definir algoritmo
algoritmo = SVC(kernel='linear')

# entrenar modelo
algoritmo.fit(x_train, y_train)
print('\nModelo entrenado')

# realizar prediccion
y_pred = algoritmo.predict(x_test)

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

# obtener + datos (ej:user)
neg_users = []
d = []
i = 0
while i < len(y_pred):
    if y_pred[i]== 'neg':
        c = x_test0.iloc[i]
        print('\nComentario negativo:\n' + c['comment'])
        if c['user'] not in neg_users:
            neg_users.append([c['user'], 1])
        else:
            for user in neg_users:
                if user[0] == c['user']:
                    user[1] += 1          
    i+=1
print('\nUsuarios que han realizado algún comentario negativo:')
print(neg_users)

# guardar modelo
joblib.dump(algoritmo, 'svm_entrenado.pkl')
print('\nModelo guardado')

#------------------------------------------------------------------------------------------------------
print()