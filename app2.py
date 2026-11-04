import os
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import numpy as np
import pandas as pd
import gradio as gr

# 1. Configurar el dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Reconstruir la arquitectura de la red neuronal (ResNetAudioClassifier)
class ResNetAudioClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(ResNetAudioClassifier, self).__init__()
        # No descargamos pesos preentrenados de ImageNet aquí porque cargaremos nuestro .pth entrenado
        self.resnet = models.resnet18(weights=None)

        # Modificar la primera capa convolucional para aceptar 1 solo canal (espectrograma)
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )

        # Modificar la capa final para las 50 clases de ESC-50
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Cargar el modelo y los pesos guardados
model = ResNetAudioClassifier(num_classes=50).to(device)
weights_path = "modelo_esc50_resnet.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("✅ ¡Modelo cargado correctamente desde el archivo .pth!")
else:
    print(f"⚠️ Advertencia: No se encontró el archivo '{weights_path}'.")

# 3. Cargar el mapeo de clases desde el archivo CSV de metadatos de ESC-50
csv_path = "ESC-50-dataset/ESC-50-master/meta/esc50.csv"
if os.path.exists(csv_path):
    df_meta = pd.read_csv(csv_path)
    clases_map = dict(zip(df_meta['target'], df_meta['category']))
else:
    clases_map = {i: f"Clase {i}" for i in range(50)}

# 4. Función de preprocesamiento e inferencia para Gradio
def clasificar_audio(audio_path):
    if audio_path is None:
        return "Por favor, graba o sube un archivo de audio."
    
    SR = 22050
    # Cargar y remuestrear el audio usando Librosa[cite: 7]
    audio, sr = librosa.load(audio_path, sr=SR)

    # Calcular el Espectrograma Mel (128 bandas Mel, hop_length=512)[cite: 7]
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Ajustar dimensiones para que coincidan con el tensor de PyTorch [1, 1, n_mels, tiempo]
    tensor_audio = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Realizar la predicción sin calcular gradientes
    with torch.no_grad():
        outputs = model(tensor_audio)
        probabilidades = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Obtener las 3 predicciones principales con sus porcentajes de certeza
        top_probs, top_indices = torch.topk(probabilidades, 3)
        
        resultados = {}
        for i in range(len(top_indices)):
            idx = top_indices[i].item()
            prob = top_probs[i].item() * 100
            nombre_clase = clases_map.get(idx, f"Clase {idx}")
            resultados[nombre_clase] = float(prob)

    return resultados

# 5. Crear la Interfaz con Gradio
demo = gr.Interface(
    fn=clasificar_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Entrada de Audio"),
    outputs=gr.Label(num_top_classes=3, label="Predicciones del Modelo (ESC-50)"),
    title="Clasificador de Sonidos Ambientales con ResNet18",
    description="Graba un sonido con tu micrófono o sube un archivo de audio (.wav/.mp3) para que la red neuronal prediga de qué categoría se trata."
)

if __name__ == "__main__":
    demo.launch(share=True)