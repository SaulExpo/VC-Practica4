## Práctica 4 Visión por Computador

### Contenidos

[Preparación](#preparación)  

[Entrenar Modelo](#entrenar-modelo)

[Detección en el vídeo](#detección-en-el-vídeo)  

[Comparativa de OCRs](#comparativa-de-ocrs)  


### Preparación

En esta práctica 4, para la preparación será necesario importar los siguientes módulos:
 - cv2: Lectura, procesamiento de vídeo, dibujado de cajas y escritura del resultado.
 - easyocr: Reconocimiento óptico de caracteres en matrículas.
 - pytesseract: Segunda alternativa OCR para comparación de rendimiento y precisión.
 - time: Medición de tiempos de inferencia para la comparativa entre OCRs.
 - csv: Exportación de los resultados (detecciones, tracking y OCR) a archivo `.csv`.
 - os: Gestión de rutas y archivos durante las ejecuciones.
 - torch: Aceleración con GPU y soporte para los modelos entrenados con YOLO.
 - ultralytics: Utilización y entrenamiento de modelos YOLO para detección y seguimiento.

### Entrenar Modelo

Para la detección de matrículas se ha utilizado un modelo \textbf{YOLO11s}, el cual ofrece un equilibrio adecuado entre precisión y velocidad, lo que lo hace apropiado para aplicaciones en vídeo. 

El entrenamiento se ha realizado sobre un conjunto de datos anotado manualmente y organizado en formato \texttt{YOLO}, definido mediante el archivo \texttt{dataset/data.yaml}, que especifica las rutas a las imágenes de \textit{train} y \textit{val}, así como la clase ``matrícula''.

Se ha llevado a cabo el entrenamiento en dos fases:

\begin{enumerate}
    \item \textbf{Entrenamiento inicial}: búsqueda de una representación general robusta, aplicando técnicas avanzadas de aumento de datos para mejorar la capacidad de generalización.
    \item \textbf{Fine-tuning}: ajuste fino reduciendo las transformaciones para especializar el modelo exclusivamente en matrículas del contexto del vídeo.
\end{enumerate}

\vspace{0.4cm}
\subsection*{Entrenamiento inicial}

Se emplearon aumentos agresivos como \textit{mosaic}, \textit{mixup} y variaciones de brillo/color. Esto permite que el modelo aprenda a detectar matrículas bajo variaciones fuertes de iluminación y perspectiva.

\begin{verbatim}
torch.cuda.empty_cache()

model = YOLO('yolo11s.pt') 

model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=960,
    batch=4,
    name='matriculas_yolo_v2',
    device='cuda',
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5, 
    translate=0.10,
    scale=0.6,  
    shear=2.0,
    flipud=0.3,
    mixup=0.1,
    mosaic=1.0
)
\end{verbatim}

\vspace{0.4cm}
\subsection*{Fine-tuning}

Posteriormente, se redujeron los aumentos para estabilizar la convergencia y mejorar la precisión en casos reales del vídeo analizado.

\begin{verbatim}
torch.cuda.empty_cache()

model = YOLO('runs/detect/matriculas_yolo_v2/weights/best.pt')

model.train(
    data='dataset/data.yaml',
    epochs=20,
    imgsz=960,
    batch=2,
    device='cuda',
    mosaic=0.0,
    mixup=0.0,
    degrees=0,
    translate=0.05,
    scale=0.4,
    shear=0,
    flipud=0.1,
    name='matriculas_yolo_v2_finetuned'
)
\end{verbatim}

\noindent
Tras el entrenamiento, el modelo final se almacenó como:

\begin{center}
\texttt{runs/detect/matriculas\_yolo\_v2\_finetuned/weights/best.pt}
\end{center}

Este modelo se utiliza posteriormente para la detección de matrículas en vídeo y para la comparación entre OCRs.


#### 1.1 Dibujar el botón

Esta función de lo único que se encarga es de obtener la imagen y unas coordenadas y sobre ella dibujar un botón con el que posteriormente se podrá hacer clic para contar el dinero

#### 1.2 Evento del clic

La segunda función desarrollada detecta las coordenadas del clic en pantalla y si dichas coordenadas coinciden dentro del botón previamente dibujado se procederá a llamar a la siguiente función que se basará en contar dinero. 

En el caso de que no se haya hecho clic en el botón, se procederá a intentar detectar sobre qué moneda se ha clicado (cuanto más cerca del centro de la moneda se clique mejor). Una vez detectada la moneda simplemente se actualizará la variable ```circulo_sel``` que se usará para calcular el total del dinero

#### 1.3 Calcular dinero

Lo primero que se plantea es crear 2 diccionarios que se usarán:
 - El primero es el diccionario ```diametro_monedas_real```: que cuenta con el diámetro de las monedas en el mundo real
 - El segundo es el diccionario ```diametro_monedas_foto```: que se inicializa a vacío para luego rellenarlo con las medidas que corresponderían a la imagen

La función empieza con un diccionario que cuenta con una string y un array para determinar el número de veces que aparece una moneda de un valor en la imagen

Posteriormente se escalarán los diametros rellenando el segundo diccionario explicado y ya no estará vacío

El siguiente paso será detectar todos los círculos de la imagen y con ellos calcular a que moneda hace referencia. Para cada uno de estos círculos una vez detectado su valor, se añadirá al diccionario para contar las monedas.

Como paso final se sumarán todas las monedas obtenidas y se creará un texto en la esquina inferior derecha de la imagen con el valor total

#### 1.4 Código principal

En este código se inicializa la imagen, se realiza un umbralizado y se aplica Canny y posteriormente Hough para detectar los círculos (los cuales se introducen en una variable) y se declaran algunas otras variables como las coordenadas del botón o la llamada del ratón para el clic.

Las siguientes imágenes son las usadas para hacer las pruebas:

![Monedas 1](Monedas.jpg)

![Monedas 2](Monedas2.jpg)

Y a continuación se podrán observar ambos resultados:

 - El primero da el resultado exacto el cual serían esos 3,88€

![Monedas Resultado](Resultado_monedas1.png)

 - El segundo detecta bien todas las monedas pero por deformaciones de la cámara o sombras lo que más se ha a podido aproximar es conseguir el siguiente resultado, muy cerca del real:

![Monedas Resultado](Resultado_monedas2.png)

### Tarea 2: Extraer Características

Al igual que la tarea anterior se ha dividido en funciones este apartado

#### 2.1 Obtener contornos

La primera función a definir es la encargada de obtener los contornos de cada uno de los fragmentos de la imagen. Para ello se realizan 6 pasos:

 1. Se suaviza la imagen para eliminar el exceso de ruido que se encuentra entre los fragmentos al pasar la imagen a grises
 2. Se utilizará Canny para detectar los bordes de los fragmentos: ```cv2.Canny(blur, 10, 40)```
 3. Tras hacer pruebas se observó que la mayoría de los bordes están mal definidos por lo que para ayudar a mejorarlos se procederá a intentar unir los bordes y dilatarlos un poco para su mejor detección a posteriori:
  - ```cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))```
  - ```cv2.dilate(edges, kernel, iterations=2)```
 4. Los fragmentos que aún tengan algún hueco serán cerrados en este paso
 5. En caso de aún así detectar algún contorno demasiado pequeño, se establece un área mínima de fragmentos indicando que los que encuentre menor a dicho área no los considere
 6. Una vez realizado todo esto se obtienen los contornos con la función ```findContours()```.

#### 2.2 Calcular métricas

Para poder obtener unas métricas para poder distinguir los fragmentos en la imagen final se han utilizado 3 imágenes con cada una un tipo de fragmento para poder analizar sus características en común. Estas son dichas imágenes:

![Fragmentos](fragment-03-olympus-10-01-2020.JPG)

![Pellet](pellet-03-olympus-10-01-2020.JPG)

![Alquitrán](tar-03-olympus-10-01-2020.JPG)

Para medir los datos se ha decidido obtener las siguientes métricas:

 - Área: Obtenida con ```cv2.contourArea()```
 - Perímetro: Obtenido con ```cv2.arcLength()```
 - Compacidad
 - Relación del área de la partícula con el contenedor
 - Relación del ancho y el alto del contenedor
 - Relación entre los ejes de la elipse ajustada
 - Relación entre las distancias menor y mayor al contorno
 - Nivel de intensidad de los píxeles
 - Nivel de circunferencia del fragmento

Todos estos datos se pasarán a un dataframe que se utilizará posteriormente

#### 2.3 Clasificación de fragmentos

El objetivo final se basa en dentro de la siguiente imagen intentar detectar a qué tipo de fragmento equivale cada uno.

![Alquitrán](MPs_test.jpg)

Para ello en esta nueva función primero asignamos un color para cada tipo de fragmento (este será el color del contorno del mismo así como de la letra de su inicial que aparecerá en la imagen final). Los colores asignados son:

 - Fragmento: Rojo
 - Piedras: Verde
 - Alquitrán: Azul

Lo siguiente es obtener las medidas previamente calculadas y además se decidió escalarlas en función de sobre todo 2 características, la intensidad de los píxeles y el grado de circularidad pues estas 2 medidas ayudan a diferenciar los fragmentos. Aparte se utilizará un clasificador de vecinos para mayor acierto

Una vez hecho todo lo anterior se procederá a calcular las métricas de cada uno de los contornos de la imagen final y se ajustarán al tipo de fragmento decidido. Tras ello se dibujará su contorno y su inicial con el texto indicado para mostrarlo sobre la imagen.

Dentro de esta función también se leen los fragmentos reales obtenidos del archivo "MPs_test_bbs.csv" para poder crear la mariz de confusión.

Gracias a una sugerencia de la inteligencia artifical, se implementó la función llamada ```calcular_iou``` que se utiliza para ayudar a detectar que etiqueta dentro del csv corresponde al fragmento calculado. Con esto, se crean 2 arrays, una llamada ```y_true``` y otra llamada ```y_pred```. Con ambas arrays se usará la función ```confusion_matrix``` y el nombre de las etiquetas ("FRA", "TAR" y "PEL") para crear la matriz cuyo resultado es el siguiente:

![Matriz de confusión](matriz_confusion.png)

#### 2.4 Código principal

En este extracto de código se definen las imágenes a analizar, algunas variables globales y se inicia desde aquí el cálculo de las métricas de las imágenes de prueba así como la clasificación final de la imagen a analizar.

El resultado obtenido, a pesar de no ser perfecto, muestra un porcentaje de acierto variado, los fragmentos y los pellets le cuesta más obtenerlos pero con los fragmentos de alquitrán tiene mayor acierto, además que detecta todos los fragmentos y no deja ninguno sin analizar. La separación de los fragmentos analizados:

![Alquitrán](Resultado_final.png)




Saúl Expósito Morales



