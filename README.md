# Clasificación de sonidos ambientales con Transfer Learning (ESC-50)

**Producto Integrador de Aprendizaje – Ciencia de Datos**
Universidad Autónoma de Nuevo León, Facultad de Ingeniería Mecánica y Eléctrica
Coordinación de Administración y Sistemas | Academia de Ciencia de Datos
Ingeniería en Inteligencia Artificial | Enero-Junio 2026
Docente: Dra. Sara Elena Garza Villarreal

## Equipo Linux

- Sergio Chiw Wah Tam Hernández
- Luis Ricardo Salinas de la Peña
- Peyton Oved Sánchez Olguín
- Francisco Gabriel Delgado Ortiz
- Alessandro Galván Junquera

---

## Resumen

Se entrenó un modelo que clasifica grabaciones de sonido ambiental en 50 categorías (perros, lluvia, sirenas, teclado, etc.). Cada audio se convierte en un espectrograma de Mel, que se trata como una imagen en escala de grises, y se clasifica con una **ResNet18 preentrenada** adaptada con PyTorch.

| | |
|---|---|
| **Tarea** | Análisis predictivo (clasificación multiclase) |
| **Dataset** | [Environmental Sound Classification 50 (ESC-50)](https://github.com/karolpiczak/ESC-50): 2,000 clips de 5 s, 50 clases, 40 clips por clase |
| **Modelo** | Transfer learning con ResNet18 (PyTorch / torchvision) sobre espectrogramas de Mel |
| **Herramienta de IA** | Gemini, como asistente de desarrollo, depuración y análisis de métricas |

## Resultados

Entrenamiento con los folds 1 a 4 de ESC-50 y validación con el fold 5, durante 100 épocas.

| Métrica | Valor |
|---|---|
| Exactitud de validación máxima | **71.75 %** (época 60) |
| Exactitud de validación al final | 69.25 % (época 100) |
| Exactitud de entrenamiento al final | 99.94 % (época 100) |

![Curva de entrenamiento](resultados/curva_entrenamiento.png)

La exactitud de validación se estabiliza alrededor del 65-70 %, mientras que la de entrenamiento llega casi al 100 %. Esa brecha indica sobreajuste, algo esperable al ajustar una red como ResNet18 con solo 1,600 clips de entrenamiento. Se redujo con aumento de datos, Dropout y weight decay, pero no desapareció.

Además, se probó la inferencia en vivo grabando 5 segundos desde el micrófono en Google Colab; en la prueba incluida en la libreta el modelo clasificó correctamente un sonido de gallo (`rooster`) con 99.84 % de certeza. Es un ejemplo ilustrativo, no una métrica de evaluación.

## Metodología

**Preprocesamiento**
1. Remuestreo de 44,100 Hz a 22,050 Hz con `librosa`.
2. Espectrograma de Mel con 128 bandas (`n_mels=128`, `hop_length=512`).
3. Conversión a decibelios (`power_to_db`).
4. Se agrega una dimensión de canal, así que cada entrada tiene forma `(1, 128, 216)`.

**Aumento de datos (solo en entrenamiento)**
- Ruido blanco aleatorio.
- SpecAugment: enmascaramiento de bandas de frecuencia y de tiempo.

**Modelo y entrenamiento**
- ResNet18 con pesos preentrenados (`weights='DEFAULT'`).
- Primera capa convolucional adaptada a 1 canal.
- Capa final reemplazada por `Dropout(0.5)` + `Linear(512, 50)`.
- Pérdida: entropía cruzada. Optimizador: Adam (`lr=0.001`, `weight_decay=1e-4`).
- Scheduler: `ReduceLROnPlateau` (factor 0.5, paciencia 5). 100 épocas.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `NotebookPIAEQLinux.ipynb` | Libreta de Google Colab: descarga y preparación de datos, entrenamiento, evaluación e inferencia en vivo |
| `EQLinux_PIA_EJ2026.docx.pdf` | Reporte escrito completo del Producto Integrador de Aprendizaje |
| `resultados/curva_entrenamiento.png` | Gráfica de exactitud por época, generada a partir de las salidas de la libreta |
| `requirements.txt` | Dependencias de Python |

## Cómo ejecutarlo

La libreta está pensada para **Google Colab con GPU**.

1. Abre `NotebookPIAEQLinux.ipynb` en [Google Colab](https://colab.research.google.com/) (Archivo → Subir cuaderno) y selecciona *Entorno de ejecución → Cambiar tipo de entorno → GPU*.
2. Ejecuta las celdas en orden. La libreta descarga ESC-50 directamente desde su repositorio oficial.
3. El entrenamiento de 100 épocas guarda los pesos en `modelo_esc50_resnet.pth`.
4. La última celda de inferencia en vivo usa `google.colab` y el micrófono del navegador, por lo que solo funciona en Colab.

Para ejecutarlo fuera de Colab, instala las dependencias con `pip install -r requirements.txt` y omite o adapta la celda de grabación.

## Referencia

K. J. Piczak. *ESC: Dataset for Environmental Sound Classification.* Proceedings of the 23rd ACM International Conference on Multimedia, 2015.
