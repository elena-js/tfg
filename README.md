# Desarrollo de un Sistema de Detección de Ciberacoso en Redes Sociales

Este proyecto tiene como resultado un sistema que, a partir de comentarios extraídos de perfiles públicos de la red social Instagram, detecta si se tratan de comentarios positivos o negativos. Para ello utiliza un algoritmo de Machine Learning supervisado, concretamente de clasificación (SVM).

## Descripción de los ficheros

- **`caradelevigne/`** -> Carpeta con el contenido del perfil `@caradelevigne`.
- **`danbilzerian/`** -> Carpeta con el contenido del perfil `@danbilzerian`.
- **`kourtneykardash/`** -> Carpeta con el contenido del perfil `@kourtneykardash`.

- **`preprocess.py`** -> Fichero correspondiente al tratamiento de los datos.

- **`sent_an.py`** -> Fichero corresponiente al sistema de detección de comentarios negativos.

- **`obtencioninfo.py`** -> Fichero correspondiente a la aplicación del sistema a casos prácticos (obtención de estadísticas de ciberacoso).

- **`emoji.csv`** -> Dataset que asocia cada emoji con un valor de sentimiento positivo, otro negativo y otro neutral (procedente de `Emoji Sentiment Ranking`).
- **`emoji_sent.py`** -> Fichero que asigna un único valor de sentimiento a cada emoji (según el contenido de `emoji.csv`).
- **`sent_emojis.csv`** -> Dataset creado a partir de `emoji_sent.py` con cada emoji y su valor de sentimiento.

