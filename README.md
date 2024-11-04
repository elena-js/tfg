# Desarrollo de un Sistema de Detección de Ciberacoso en Redes Sociales

Este proyecto tiene como resultado un sistema que, a partir de comentarios extraídos de perfiles públicos de la red social Instagram, detecta si se tratan de comentarios positivos o negativos. Para ello utiliza un algoritmo de Machine Learning supervisado, concretamente de clasificación (SVM).

## Lista de Contenido
- [Estructura del proyecto](#estructura-del-proyecto)
- [Descripción de los ficheros](#descripción-de-los-ficheros)
- [Instalación](#instalación)
- [Uso](#uso)
  
## Estructura del proyecto

## Descripción de los ficheros

- **caradelevigne** -> Carpeta con contenido del perfil @caradelevigne
- **danbilzerian** -> Carpeta con contenido del perfil @danbilzerian
- **kourtneykardash** -> Carpeta con contenido del perfil @kourtneykardash

- **preprocess.py** -> Tratamiento de los datos

- **emoji.csv** -> Dataset de emoji sentiment ranking
- **emoji_sent.py** -> Asigna valor de sentimiento a cada emoji
- **sent_emojis.csv** -> CSV creado a partir de emoji_sent.py

- **sent_an.py** -> Sistema detección comentarios negativos

- **obtencioninfo.py** -> Aplicación del sistema a casos prácticos (obtención estadísticas de ciberacoso)

## Instalación

## Uso
