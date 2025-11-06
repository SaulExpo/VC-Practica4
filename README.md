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

Para la detección de matrículas se ha utilizado como punto de partida el modelo `YOLO11s`, que ofrece un buen equilibrio entre precisión y velocidad.  
Este modelo se entrenó utilizando un conjunto de imágenes propio, anotado previamente y organizado en formato YOLO.

El dataset cuenta con imágenes de matrículas tomadas desde distintos ángulos, distancias y condiciones de iluminación, como se muestra a continuación:

<table align="center">
  <tr>
    <td><img src="assets/img1.jpg" width="120"/></td>
    <td><img src="assets/img2.jpg" width="120"/></td>
    <td><img src="assets/img3.jpg" width="120"/></td>
    <td><img src="assets/img4.jpg" width="120"/></td>
    <td><img src="assets/img5.jpg" width="120"/></td>
    <td><img src="assets/img6.jpg" width="120"/></td>
    <td><img src="assets/img7.jpg" width="120"/></td>
  </tr>
</table>

#### Entrenamiento en dos fases

##### **1) Entrenamiento inicial**
Se entrenó el modelo aplicando técnicas de *data augmentation* para mejorar su capacidad de generalización:

| Parámetro | Descripción |
|-----------|-------------|
| `mosaic` | Combina 4 imágenes distintas en una sola, cambiando contexto y fondo. Esto ayuda a detectar matrículas en posiciones poco comunes. |
| `mixup`  | Superpone parcialmente dos imágenes mezclando sus píxeles. Esto obliga al modelo a ser más robusto ante solapamientos y ruido. |
| `degrees` | Rotación aleatoria de la imagen para aumentar variabilidad. |
| `scale` | Zoom aleatorio (acercar / alejar). |
| `shear` | Deforma la imagen simulando perspectiva. |
| `flipud` | Invierte verticalmente algunas imágenes. |

Este entrenamiento produce un modelo generalista capaz de detectar matrículas en múltiples situaciones.

##### **2) Fine-tuning**
En esta fase se desactivan la mayoría de aumentos y se ajusta el modelo específicamente a las condiciones del vídeo final (ángulo, color, distancia), mejorando la precisión real.

 - Segunda: ajuste fino reduciendo las transformaciones para especializar el modelo exclusivamente en matrículas del contexto del vídeo.

#### Modelo Final

Al terminar la segunda fase, el modelo resultante se almacenó como:
`matriculas_yolo_v2_finetuned`
Este modelo se utiliza posteriormente para la detección de matrículas en vídeo.

En la siguiente gráfica se puede observar como tanto la train_loss como la val_boss descienden a la vez lo que indica que el entrenamiento fue exitoso. Además se ve que el porcentaje de hacierto superó el 90% lo que ayuda a recalcar la gran eficacia de entrenamiento del modelo

<img width="1594" height="795" alt="image" src="https://github.com/user-attachments/assets/a4d8d3de-5b25-47b8-aede-32c97f0133b2" />


### Detección en el vídeo

Una vez entrenado el modelo, se procede al procesamiento del vídeo para realizar:

- Detección de personas y vehículos.
- Seguimiento (tracking) de cada objeto mediante identificadores.
- Detección y reconocimiento de matrículas.
- Ocultar el rostro de las personas para hacerlas anónimas.
- Guardar el video del resultado final.
- Generar un archivo `.csv` con las comparativas.

Para ello se utilizan dos modelos YOLO diferentes:

| Modelo | Función |
|--------|---------|
| `yolo11s.pt` | Detecta y sigue personas y vehículos. |
| `matriculas_yolo_v2_finetuned.pt` | Modelo entrenado en el paso anterior para la detección de matrículas. |

Además, se emplea `EasyOCR` usado para leer el texto de la matrícula.



#### **Seguimiento de objetos (Tracking)**

El seguimiento se realiza mediante el algoritmo **ByteTrack**, lo que permite asignar un identificador único a cada objeto detectado y mantenerlo a lo largo del vídeo.  
De esta forma, se evita contar el mismo vehículo o persona más de una vez.
Cada clase posee su propio contador independiente:
 - Person (personas)
 - Car (coches)
 - Truck (Camiones)
 - Motorcycle (motos)
 - Bus (Guaguas)

Además, se asigna un código de identificación a cada clase

#### **Anonimización de personas**

Para respetar la privacidad, se aplica un **pixelado automático** en la parte superior de los cuadros de detección de personas (se pixela la parte superior asumiendo que tapa la cara de la persona).  

#### **Detección y lectura de matrículas**

Solo en los objetos clasificados como vehículos (coche, autobús, camión, moto) se recorta la región, se le dibuja un cuadrado amarillo al rededor de la matrícula y se envía a EasyOCR para que sobre el video aparezca lo que detecta que pone.  
#### **Generación de resultados**

El sistema produce dos archivos:

| Archivo | Descripción |
|---------|-------------|
| `resultado_final.mp4` | Vídeo original con detecciones, seguimiento, matrículas y anonimización. |
| `detecciones_tracking.csv` | Datos de cada detección con coordenadas, ID y texto de la matrícula. |

Ejemplo de línea del CSV:

| frame | tipo_objeto | conf | track_id | x1 | y1 | x2 | y2 | conf_matricula | mx1 | my1 | mx2 | my2 | texto_matricula |
|-------|-------------|------|----------|----|----|----|----|----------------|-----|-----|-----|-----|-----------------|
| 84 | car | 0.87 | C-3 | 651 | 310 | 732 | 360 | 0.91 | 675 | 338 | 720 | 352 | "1770 JYG" |

Fragmento del video: 

<img width="3840" height="1080" alt="image" src="assets/gif1.gif" />



### Comparativa de OCRs

Para evaluar el rendimiento de dos modelos de OCR se han comparado para analizarlos:

| OCR | Descripción | Ventajas | Inconvenientes |
|-----|-------------|----------|----------------|
| **EasyOCR** | Modelo basado en Deep Learning | Buen rendimiento en imágenes reales y con ruido | Más lento |
| **Tesseract** | Motor OCR clásico basado en reglas | Muy rápido y ligero | Sensible al ruido e iluminación, peor precisión |

#### Metodología de evaluación

Se seleccionaron **50 imágenes de matrículas** recortadas del dataset hasta quedarse solo con la matrícula en sí en primer plano para que pueda ser fácilmente reconocida.

Para cada imagen se midió:

- **Texto reconocido por EasyOCR**
- **Texto reconocido por Tesseract**
- **Tiempo de inferencia (ms)**
- **Acierto** respecto a la matrícula real anotada manualmente.

El proceso se automatizó generando un archivo CSV con el siguiente formato:

| imagen | matricula_real | easyocr_texto | tesseract_texto | acierto_easyocr | tiempo_easyocr(ms) | tiempo_tesseract(ms) |
|--------|----------------|---------------|-----------------|-----------------|--------------------|----------------------|

#### Resultados

A partir de las 50 imágenes de matrículas analizadas se han obtenido los siguientes promedios:

| Métrica | EasyOCR | Tesseract |
|--------|--------|-----------|
| **Precisión de lectura** | **≈ 30–40%** (dependiendo de la imagen) | **≈ 0–5%** |
| **Tiempo medio (ms)** | ~ **50–100 ms** | ~ **180–210 ms** |

> **EasyOCR obtiene una precisión mayor**.
>
> **Tesseract falla en casi todas las matrículas**.

---

#### Visualización de resultados

A continuación se muestran dos gráficas comparativas obtenidas en GoogleSheets:

<p align="center">
  <img width="450" alt="image" src="https://github.com/user-attachments/assets/560b57c8-54d1-4e8d-8709-16b6386db9f7" />
</p>

<p align="center">
  <em>Figura 1: Porcentaje de acierto de EasyOCR vs Tesseract</em>
</p>

<p align="center">
  <img width="450" alt="image" src="https://github.com/user-attachments/assets/cce56033-0721-40bb-af42-5f4a14c06a87" />

</p>

<p align="center">
  <em>Figura 2: Tiempo de inferencia medio por OCR</em>
</p>

---

#### Conclusiones

- EasyOCR **reconoce matrículas de manera más robusta**, incluso en movimiento o baja calidad.
- Tesseract **no es adecuado** para matrículas de vídeo real sin un preprocesado muy agresivo.
- El tiempo adicional de EasyOCR **compensa su mejora en precisión**.
- Por tanto, **EasyOCR** es la opción utilizada para el sistema final integrado en el vídeo.

#### Extra

Para acceder a todas las carpetas con el dataset, los videos y el modelo entrenado se usará el siguiente enlace a una carpeta de Google Drive:

https://drive.google.com/drive/folders/1tRtH5754omyO210dmpvrD-rsF6KSAAMZ?usp=sharing

Saúl Expósito Morales



