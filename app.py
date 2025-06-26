import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
import torch.nn as nn
import matplotlib.colors as mcolors

# Import Plotly pentru funcționalitatea hover
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly nu este instalat! Funcționalitatea hover nu va fi disponibilă. Rulează: pip install plotly")

# Importurile pentru modelul tău
try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    st.error("❌ segmentation_models_pytorch nu este instalat! Rulează: pip install segmentation_models_pytorch")

# Configurare pagină - optimizat pentru spațiu
st.set_page_config(
    page_title="Demo Segmentare Hiperspectrale",
    page_icon="🌱",
    layout="wide"
)

# ===================================================================
# MAPAREA CLASELOR UAV-HSI CU CULORI OPTIMIZATE
# ===================================================================

UAV_HSI_CLASSES = {
    0: {"name_en": "NULL", "name_ro": "Fundal/Nedefinit", "color": "#000000"},  # Negru
    1: {"name_en": "SollWeed", "name_ro": "Buruieni de sol", "color": "#8B4513"},  # Maro
    2: {"name_en": "Chin_Cabbage", "name_ro": "Varză chinezească", "color": "#90EE90"},  # Verde deschis
    3: {"name_en": "Millet", "name_ro": "Mei", "color": "#F0E68C"},  # Khaki
    4: {"name_en": "Leaf_mustard", "name_ro": "Muștar frunze", "color": "#ADFF2F"},  # Verde-galben
    5: {"name_en": "Greenbead", "name_ro": "Mazăre verde", "color": "#228B22"},  # Verde pădure
    6: {"name_en": "Spinach", "name_ro": "Spanac", "color": "#006400"},  # Verde închis
    7: {"name_en": "Bok_Choy", "name_ro": "Bok choy", "color": "#98FB98"},  # Verde pal
    8: {"name_en": "Turnip", "name_ro": "Nap", "color": "#E6E6FA"},  # Lavandă
    9: {"name_en": "Cotton", "name_ro": "Bumbac", "color": "#F5F5DC"},  # Bej
    10: {"name_en": "Corn", "name_ro": "Porumb", "color": "#FFD700"},  # Auriu
    11: {"name_en": "Carrot", "name_ro": "Morcov", "color": "#FF8C00"},  # Portocaliu închis
    12: {"name_en": "Sorghum", "name_ro": "Sorg", "color": "#CD853F"},  # Peru
    13: {"name_en": "Pumpkin", "name_ro": "Dovleac", "color": "#FF6347"},  # Roșu-portocaliu
    14: {"name_en": "Kohlrabi", "name_ro": "Gulie", "color": "#DDA0DD"},  # Mov
    15: {"name_en": "Scallion", "name_ro": "Ceapă verde", "color": "#32CD32"},  # Verde lime
    16: {"name_en": "Sweet_potato", "name_ro": "Cartof dulce", "color": "#A0522D"},  # Maro sienna
    17: {"name_en": "Peanut", "name_ro": "Arahide", "color": "#DEB887"},  # Burlywood
    18: {"name_en": "Sesame", "name_ro": "Susan", "color": "#F4A460"},  # Sandy brown
    19: {"name_en": "Beans", "name_ro": "Fasole", "color": "#800000"},  # Maro roșcat
    20: {"name_en": "Road", "name_ro": "Drum", "color": "#696969"},  # Gri închis
    21: {"name_en": "Tobacco", "name_ro": "Tutun", "color": "#B8860B"},  # Galben închis
    22: {"name_en": "Herbs", "name_ro": "Ierburi aromatice", "color": "#9ACD32"},  # Verde-galben
    23: {"name_en": "Cauliflower", "name_ro": "Conopidă", "color": "#FFFAF0"},  # Alb floral
    24: {"name_en": "Eggplant", "name_ro": "Vânătă", "color": "#4B0082"},  # Indigo
    25: {"name_en": "Daikon", "name_ro": "Ridiche albă", "color": "#F0F8FF"},  # Alice blue
    26: {"name_en": "Pepper", "name_ro": "Ardei", "color": "#FF0000"},  # Roșu
    27: {"name_en": "Multiflecte", "name_ro": "Culturi mixte", "color": "#FF1493"},  # Deep pink
    28: {"name_en": "Tree", "name_ro": "Copaci", "color": "#654321"},  # Maro închis
    29: {"name_en": "Okra", "name_ro": "Bame", "color": "#7CFC00"}  # Verde lawns
}


def create_uav_hsi_colormap():
    """Creează harta de culori optimizată pentru clasele UAV-HSI"""
    colors = [UAV_HSI_CLASSES[i]["color"] for i in range(30)]
    return mcolors.ListedColormap(colors)


def create_plotly_colorscale():
    """Creează colorscale pentru Plotly din culorile UAV-HSI"""
    colors = [UAV_HSI_CLASSES[i]["color"] for i in range(30)]

    # Creează colorscale pentru Plotly (trebuie să fie între 0 și 1)
    colorscale = []
    for i, color in enumerate(colors):
        colorscale.append([i / 29, color])  # 29 pentru că avem 30 clase (0-29)

    return colorscale


def create_interactive_segmentation_plot(data, title, width=None, height=None):
    """
    Creează un plot Plotly interactiv pentru imaginile de segmentare cu hover tooltips
    """
    if not PLOTLY_AVAILABLE:
        return None

    # Creează matricea de hover text cu informații despre clase
    hover_text = np.empty(data.shape, dtype=object)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            class_id = int(data[i, j])
            if class_id < 30:
                class_info = UAV_HSI_CLASSES[class_id]
                hover_text[i, j] = (
                    f"<b>{class_info['name_ro']}</b><br>"
                    f"Clasă: {class_id}<br>"
                    f"Nume EN: {class_info['name_en']}<br>"
                    f"Culoare: {class_info['color']}<br>"
                    f"Poziție: ({i}, {j})"
                )
            else:
                hover_text[i, j] = f"Clasă necunoscută: {class_id}<br>Poziție: ({i}, {j})"

    # Creează figura Plotly
    fig = go.Figure(data=go.Heatmap(
        z=data,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale=create_plotly_colorscale(),
        zmin=0,
        zmax=29,
        showscale=False  # Scoatem colorbar-ul pentru a avea imagini de aceeași mărime
    ))

    # FORȚEZ EXACT ACELEAȘI DIMENSIUNI CA RGB - FĂRĂ MARGINI
    data_height, data_width = data.shape

    fig.update_layout(
        title="",  # Fără titlu
        xaxis=dict(
            showticklabels=False,
            title="",
            scaleanchor="y",  # Leagă scale-ul X de Y
            scaleratio=1,  # Raport 1:1 pentru pixeli pătrați
            showgrid=False,  # Ascunde grid-ul X pentru aspect mai curat
            range=[0, data_width],  # FORȚEZ dimensiunea exactă
            constrain='domain',  # Constrânge la domeniul disponibil
        ),
        yaxis=dict(
            showticklabels=False,
            title="",
            autorange='reversed',
            constrain='domain',  # Constrânge la domeniul disponibil
            showgrid=False,  # Ascunde grid-ul Y pentru aspect mai curat
            range=[0, data_height],  # FORȚEZ dimensiunea exactă
        ),
        margin=dict(l=0, r=0, t=0, b=0),  # ZERO MARGINI PENTRU ALINIERE PERFECTĂ
        autosize=True,
        height=400,  # REDUS LA 400 PENTRU ALINIERE MAI BUNĂ CU TITLURILE
        width=None,  # Lăsam Streamlit să decidă lățimea
        # Configurez toolbar-ul să fie mai puțin intruziv
        modebar=dict(
            orientation="h",  # Orizontal
            bgcolor="rgba(50,50,50,0.7)",  # Fundal mai închis și transparent
            color="white",
            activecolor="lightblue",
            add=['drawline', 'drawopenpath']  # Adaug doar funcții utile
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Fundal transparent pentru plot
        paper_bgcolor='rgba(0,0,0,0)',  # Fundal transparent pentru întreaga figură
    )

    return fig


def create_rgb_plotly_display(rgb_data, title=""):
    """
    Creează un plot Plotly pentru imaginea RGB composit - FORȚEZ DIMENSIUNI IDENTICE
    """
    if not PLOTLY_AVAILABLE:
        return None

    # Folosesc px.imshow dar cu constrangeri EXACTE ca GT
    fig = px.imshow(rgb_data)

    # FORȚEZ EXACT ACELEAȘI DIMENSIUNI CA GT - FĂRĂ MARGINI
    height, width = rgb_data.shape[:2]

    fig.update_layout(
        title="",  # Fără titlu
        xaxis=dict(
            showticklabels=False,
            title="",
            scaleanchor="y",  # IDENTIC cu GT
            scaleratio=1,  # IDENTIC cu GT
            showgrid=False,  # IDENTIC cu GT
            range=[0, width],  # FORȚEZ dimensiunea exactă
            constrain='domain',  # IDENTIC cu GT
        ),
        yaxis=dict(
            showticklabels=False,
            title="",
            autorange='reversed',  # IDENTIC cu GT
            constrain='domain',  # IDENTIC cu GT
            showgrid=False,  # IDENTIC cu GT
            range=[0, height],  # FORȚEZ dimensiunea exactă
        ),
        margin=dict(l=0, r=0, t=0, b=0),  # ZERO MARGINI - IDENTIC CU GT
        autosize=True,
        height=400,  # IDENTIC CU GT - 400 PENTRU ALINIERE
        width=None,  # Lăsam Streamlit să decidă lățimea
        modebar=dict(
            orientation="h",
            bgcolor="rgba(50,50,50,0.7)",
            color="white",
            activecolor="lightblue",
            add=['drawline', 'drawopenpath']
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # IDENTIC cu GT
        paper_bgcolor='rgba(0,0,0,0)',  # IDENTIC cu GT
    )

    return fig


def create_matplotlib_fallback(data, title):
    """
    Fallback la matplotlib dacă Plotly nu este disponibil
    """
    fig, ax = plt.subplots(figsize=(3.6, 2.9))  # Redus ~30% pentru 100% zoom

    custom_cmap = create_uav_hsi_colormap()

    im = ax.imshow(data,
                   cmap=custom_cmap,
                   vmin=0,
                   vmax=29,
                   aspect='equal')

    ax.set_title(title, fontsize=12)
    ax.axis('off')

    # Colorbar cu denumiri
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=25, pad=0.1)
    cbar.set_label('Clase', fontsize=10)

    # Setează tick-urile cu denumiri
    unique_classes = np.unique(data)
    if len(unique_classes) <= 10:
        cbar.set_ticks(unique_classes)
        tick_labels = [f"{cl}: {UAV_HSI_CLASSES[cl]['name_ro'][:8]}..."
                       if len(UAV_HSI_CLASSES[cl]['name_ro']) > 8
                       else f"{cl}: {UAV_HSI_CLASSES[cl]['name_ro']}"
                       for cl in unique_classes if cl < 30]
        cbar.set_ticklabels(tick_labels, fontsize=7)
    else:
        cbar.set_ticks(range(0, 30, 5))
        cbar.set_ticklabels([f"{i}" for i in range(0, 30, 5)])

    plt.tight_layout()
    return fig


def get_class_names_romanian():
    """Returnează numele claselor în română"""
    return [UAV_HSI_CLASSES[i]["name_ro"] for i in range(30)]


def create_legend_figure():
    """Creează o figură separată pentru legenda claselor"""
    fig, ax = plt.subplots(figsize=(2, 12))
    ax.axis('off')

    # Creează colorbar vertical pentru legendă
    custom_cmap = create_uav_hsi_colormap()
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=29))
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=1.0, aspect=30)
    cbar.set_label('Tipuri de culturi', fontsize=14, rotation=270, labelpad=25)

    # Setează tick-urile la mijlocul fiecărei culori
    tick_positions = np.arange(0, 30)
    cbar.set_ticks(tick_positions)

    # Nume scurte pentru afișare
    short_names = [UAV_HSI_CLASSES[i]["name_ro"][:12] + "..."
                   if len(UAV_HSI_CLASSES[i]["name_ro"]) > 15
                   else UAV_HSI_CLASSES[i]["name_ro"]
                   for i in range(30)]

    cbar.set_ticklabels(short_names, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig


# ===================================================================
# CLASA MODELULUI TĂU PENTRU SEGMENTARE HIPERSPECTRALĂ
# ===================================================================

class ImprovedHyperspectralUNet(nn.Module):
    """UNet hiperspectral îmbunătățit - ADAPTAT din codul tău"""

    def __init__(self, n_bands=200, n_classes=30, encoder_name='resnet34', dropout_rate=0.2):
        super(ImprovedHyperspectralUNet, self).__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.encoder_name = encoder_name

        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch nu este disponibil!")

        # Maparea căilor către primul strat convoluțional pentru fiecare encoder
        self.conv_paths = {
            'resnet34': 'conv1',
            'resnext50_32x4d': 'conv1',
            'se_resnet50': 'layer0[0]',
            'efficientnet-b3': '_conv_stem',
            'timm-efficientnet-b3': 'conv_stem',
            'densenet121': 'features[0]',
            'mobilenet_v2': 'features.0.0',
            'timm-regnetx_004': 'model.stem.conv'
        }

        # Creează modelul UNet de bază cu 3 canale
        try:
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=3,
                classes=n_classes,
            )
        except Exception as e:
            # Fallback la ResNet34
            self.model = smp.Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=n_classes,
            )
            self.encoder_name = 'resnet34'

        # Adaptează primul strat pentru date hiperspectrale
        self._adapt_first_layer()

        # Adaugă dropout în decoder dacă specificat
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)

    def _adapt_first_layer(self):
        """Adaptează primul strat convoluțional pentru date hiperspectrale"""
        conv_path = self.conv_paths.get(self.encoder_name, 'conv1')

        try:
            # Găsește primul strat convoluțional
            first_conv = self._get_first_conv_layer(conv_path)

            if first_conv is None:
                return

            # Salvează ponderile originale
            original_weights = first_conv.weight.data.clone()

            # Creează noul strat convoluțional
            new_conv = nn.Conv2d(
                self.n_bands,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )

            # Inițializează ponderile pentru canalele suplimentare
            with torch.no_grad():
                for i in range(self.n_bands):
                    channel_idx = i % 3  # Repetă ponderile RGB
                    new_conv.weight[:, i] = original_weights[:, channel_idx]

            # Înlocuiește primul strat
            self._set_first_conv_layer(conv_path, new_conv)

        except Exception as e:
            print(f"Eroare la adaptarea primului strat: {e}")

    def _get_first_conv_layer(self, conv_path):
        """Obține primul strat convoluțional pe baza căii specificate"""
        encoder = self.model.encoder

        try:
            if conv_path == 'conv1':
                return encoder.conv1
            elif conv_path == '_conv_stem':
                return encoder._conv_stem
            elif conv_path == 'conv_stem':
                return encoder.conv_stem
            elif conv_path == 'layer0[0]':
                return encoder.layer0[0]
            elif conv_path == 'features[0]':
                return encoder.features[0]
            elif conv_path == 'features.0.0':
                return encoder.features[0][0]
            elif conv_path == 'model.stem.conv':
                return encoder.model.stem.conv
            else:
                # Încearcă să parseze calea dinamică
                parts = conv_path.split('.')
                current = encoder
                for part in parts:
                    if '[' in part and ']' in part:
                        attr_name = part.split('[')[0]
                        index = int(part.split('[')[1].split(']')[0])
                        current = getattr(current, attr_name)[index]
                    else:
                        current = getattr(current, part)
                return current
        except Exception:
            return None

    def _set_first_conv_layer(self, conv_path, new_conv):
        """Setează primul strat convoluțional pe baza căii specificate"""
        encoder = self.model.encoder

        try:
            if conv_path == 'conv1':
                encoder.conv1 = new_conv
            elif conv_path == '_conv_stem':
                encoder._conv_stem = new_conv
            elif conv_path == 'conv_stem':
                encoder.conv_stem = new_conv
            elif conv_path == 'layer0[0]':
                encoder.layer0[0] = new_conv
            elif conv_path == 'features[0]':
                encoder.features[0] = new_conv
            elif conv_path == 'features.0.0':
                encoder.features[0][0] = new_conv
            elif conv_path == 'model.stem.conv':
                encoder.model.stem.conv = new_conv
            else:
                # Gestionează căile dinamice
                parts = conv_path.split('.')
                current = encoder
                for part in parts[:-1]:
                    if '[' in part and ']' in part:
                        attr_name = part.split('[')[0]
                        index = int(part.split('[')[1].split(']')[0])
                        current = getattr(current, attr_name)[index]
                    else:
                        current = getattr(current, part)

                # Setează ultimul atribut
                last_part = parts[-1]
                if '[' in last_part and ']' in last_part:
                    attr_name = last_part.split('[')[0]
                    index = int(last_part.split('[')[1].split(']')[0])
                    getattr(current, attr_name)[index] = new_conv
                else:
                    setattr(current, last_part, new_conv)

        except Exception:
            pass

    def forward(self, x):
        """Forward pass prin rețea"""
        # Forward pass prin encoder
        features = self.model.encoder(x)

        # Forward pass prin decoder
        decoder_output = self.model.decoder(features)

        # Aplică dropout dacă este activ
        if hasattr(self, 'dropout') and self.training:
            decoder_output = self.dropout(decoder_output)

        # Segmentare finală
        masks = self.model.segmentation_head(decoder_output)

        return masks


# ===================================================================
# FUNCȚII HELPER
# ===================================================================

def get_available_models():
    """Scanează folderul models/ pentru fișiere .pth disponibile"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return {}

    available_models = {}

    # Configurațiile modelelor tale
    model_configs = {
        'DenseNet121 - Experimentul 5 (200 benzi)': {
            'filename': 'DenseNet121(exp5).pth',
            'n_bands': 200,
            'n_classes': 30,
            'encoder_name': 'densenet121',
            'dropout_rate': 0.3,
            'description': 'DenseNet121 din Experimentul 5 cu toate benzile - Dense connections pentru redundanța spectrală'
        },
        'ResNet34 - Experimentul 5 (200 benzi)': {
            'filename': 'ResNet34(exp5).pth',
            'n_bands': 200,
            'n_classes': 30,
            'encoder_name': 'resnet34',
            'dropout_rate': 0.3,
            'description': 'ResNet34 din Experimentul 5 - Arhitectura de referință stabilă'
        },
        'SE-ResNet50 - Experimentul 5 (200 benzi)': {
            'filename': 'SE_ResNet50(exp5).pth',
            'n_bands': 200,
            'n_classes': 30,
            'encoder_name': 'se_resnet50',
            'dropout_rate': 0.3,
            'description': 'SE-ResNet50 cu atenție pe canale pentru selecția benzilor spectrale'
        },
        'EfficientNet-B3 - Experimentul 5 (200 benzi)': {
            'filename': 'EfficientNet_B3(exp5).pth',
            'n_bands': 200,
            'n_classes': 30,
            'encoder_name': 'efficientnet-b3',
            'dropout_rate': 0.3,
            'description': 'EfficientNet-B3 - Scalare optimizată și eficiență computațională'
        },
        'Model Generic - Experimentul 5': {
            'filename': 'best_model.pth',
            'n_bands': 200,
            'n_classes': 30,
            'encoder_name': 'resnet34',
            'dropout_rate': 0.3,
            'description': 'Cel mai bun model din Experimentul 5'
        }
    }

    for model_name, config in model_configs.items():
        model_path = os.path.join(models_dir, config['filename'])
        if os.path.exists(model_path):
            available_models[model_name] = {
                'path': model_path,
                'config': config
            }

    return available_models


@st.cache_resource
def load_model(model_path, model_config):
    """Încarcă modelul salvat cu configurația specificată"""
    if not os.path.exists(model_path):
        return None, f"Fișierul model nu există: {model_path}"

    if not SMP_AVAILABLE:
        return None, "segmentation_models_pytorch nu este instalat"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Creează instanța modelului
        model = ImprovedHyperspectralUNet(
            n_bands=model_config['n_bands'],
            n_classes=model_config['n_classes'],
            encoder_name=model_config['encoder_name'],
            dropout_rate=model_config.get('dropout_rate', 0.2)
        )

        # Încarcă state_dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Setează modelul în modul de evaluare
        model.eval()
        model.to(device)

        return model, None

    except Exception as e:
        return None, f"Eroare la încărcarea modelului: {str(e)}"


def preprocess_hyperspectral_image(image_data):
    """Preprocessează imaginea hiperspectrală pentru inferență - EXACT ca în codul tău"""
    # Verifică dimensiunile și convertește la CHW dacă e nevoie
    if len(image_data.shape) == 3:
        if image_data.shape[2] > image_data.shape[0]:  # HWC format
            image_data = np.transpose(image_data, (2, 0, 1))  # Convertește la CHW

    # Normalizează fiecare bandă individual (ca în HyperspectralDataset)
    processed_image = image_data.astype(np.float32)
    for band_idx in range(processed_image.shape[0]):
        band = processed_image[band_idx]
        min_val, max_val = np.min(band), np.max(band)
        if max_val > min_val:
            processed_image[band_idx] = (band - min_val) / (max_val - min_val)

    # Convertește la tensor PyTorch și adaugă dimensiunea batch
    tensor_image = torch.from_numpy(processed_image).unsqueeze(0)  # (1, C, H, W)

    return tensor_image


def apply_model_inference(model, image_tensor):
    """Aplică modelul pentru inferență"""
    if model is None:
        return None, "Model nu este încărcat"

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    try:
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor)

            # Convertește la predicții (argmax)
            predictions = torch.argmax(outputs, dim=1)

            # Convertește înapoi la numpy și elimină dimensiunea batch
            predictions_np = predictions.cpu().numpy()[0]  # (H, W)

            # Probabilitățile pentru fiecare clasă (opțional)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # (C, H, W)

            return predictions_np, None

    except Exception as e:
        return None, f"Eroare la inferență: {str(e)}"


@st.cache_data
def load_available_images():
    """Încarcă lista imaginilor disponibile din folderul rs/"""
    rs_dir = "sample_data/rs/"
    if os.path.exists(rs_dir):
        files = [f for f in os.listdir(rs_dir) if f.endswith('.npy')]
        return sorted(files)
    return []


def load_hyperspectral_image(filename):
    """Încarcă imaginea hiperspectrală"""
    rs_path = f"sample_data/rs/{filename}"
    gt_path = f"sample_data/gt/{filename}"

    if os.path.exists(rs_path):
        rs_data = np.load(rs_path)
        gt_data = None
        if os.path.exists(gt_path):
            gt_data = np.load(gt_path)
        return rs_data, gt_data
    return None, None


def create_rgb_composite(hyperspectral_data):
    """Creează compositul RGB din datele hiperspectrale - EXACT din codul original"""
    if len(hyperspectral_data.shape) == 3:
        height, width, n_bands = hyperspectral_data.shape

        # Convertesc la format channels-first pentru a folosi exact codul original
        hyperspectral_image = hyperspectral_data.transpose(2, 0, 1)  # (canale, înălțime, lățime)

        # EXACT din codul original - metoda natural_colors
        # Pentru 200 de benzi, aproximăm spectrul 400-1000nm
        # RGB vizibil: ~400-700nm, deci primele ~60% din benzi
        visible_bands = int(n_bands * 0.6)

        # Selectează benzi din spectrul vizibil pentru RGB
        blue_idx = int(visible_bands * 0.2)  # ~450nm albastru
        green_idx = int(visible_bands * 0.5)  # ~550nm verde
        red_idx = int(visible_bands * 0.8)  # ~650nm roșu

        red_band = hyperspectral_image[red_idx]
        green_band = hyperspectral_image[green_idx]
        blue_band = hyperspectral_image[blue_idx]

        # Combină benzile direct fără modificări complexe (ca în original)
        red_channel = red_band
        green_channel = green_band
        blue_channel = blue_band

        # Combină canalele
        rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=2)

        # EXACT normalizarea din codul original
        for i in range(3):
            channel = rgb_image[:, :, i]

            # Folosește percentile pentru a elimina valorile extreme
            p1, p99 = np.percentile(channel, (1, 99))
            channel_clipped = np.clip(channel, p1, p99)

            # Normalizează între 0 și 1
            if p99 > p1:
                rgb_image[:, :, i] = (channel_clipped - p1) / (p99 - p1)
            else:
                rgb_image[:, :, i] = np.zeros_like(channel)

        # Asigură-te că valorile sunt în [0,1]
        rgb_image = np.clip(rgb_image, 0, 1)

        return rgb_image
    return None


def calculate_ndvi(hyperspectral_data):
    """Calculează NDVI din datele hiperspectrale"""
    if len(hyperspectral_data.shape) == 3:
        h, w, bands = hyperspectral_data.shape

        if bands >= 150:
            nir_band = int(bands * 0.8)  # ~800nm
            red_band = int(bands * 0.5)  # ~650nm
        else:
            nir_band = min(bands - 1, 80)
            red_band = min(bands - 1, 50)

        nir = hyperspectral_data[:, :, nir_band].astype(float)
        red = hyperspectral_data[:, :, red_band].astype(float)

        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = np.zeros_like(nir)
        valid_mask = (nir + red) > 0
        ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / (nir[valid_mask] + red[valid_mask])

        return ndvi
    return None


def calculate_evi(hyperspectral_data):
    """Calculează EVI (Enhanced Vegetation Index)"""
    if len(hyperspectral_data.shape) == 3:
        h, w, bands = hyperspectral_data.shape

        if bands >= 150:
            nir_band = int(bands * 0.8)
            red_band = int(bands * 0.5)
            blue_band = int(bands * 0.2)
        else:
            nir_band = min(bands - 1, 80)
            red_band = min(bands - 1, 50)
            blue_band = min(bands - 1, 20)

        nir = hyperspectral_data[:, :, nir_band].astype(float)
        red = hyperspectral_data[:, :, red_band].astype(float)
        blue = hyperspectral_data[:, :, blue_band].astype(float)

        # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        denominator = nir + 6 * red - 7.5 * blue + 1
        evi = np.zeros_like(nir)
        valid_mask = denominator != 0
        evi[valid_mask] = 2.5 * (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]

        return evi
    return None


# ===================================================================
# UI PRINCIPAL
# ===================================================================

# UI Principal - Compactat
st.markdown("## 🌱 Demo Segmentare UAV-HSI")  # Micșorez titlul
st.markdown("**Interfață demonstrativă pentru segmentarea semantică a culturilor agricole**")

# Sidebar pentru controluri - compactat
st.sidebar.header("⚙️ Setări")

# 1. Load Model - compactat
st.sidebar.subheader("🔧 Model")

# Scanează modelele disponibile
available_models = get_available_models()

if available_models:
    model_choice = st.sidebar.selectbox(
        "Model:",
        list(available_models.keys()),
        help="Alege modelul antrenat"
    )

    # Informații model compacte - fără expander
    if model_choice in available_models:
        model_info = available_models[model_choice]
        config = model_info['config']
        st.sidebar.caption(f"**{config['encoder_name']}** | {config['n_bands']} benzi | {config['n_classes']} clase")

    if st.sidebar.button("🔄 Încarcă"):
        with st.spinner("Se încarcă..."):
            model_info = available_models[model_choice]
            model, error = load_model(model_info['path'], model_info['config'])

            if model is not None:
                st.session_state.model = model
                st.session_state.model_config = model_info['config']
                st.session_state.model_name = model_choice
                st.sidebar.success("✅ Încărcat!")

                # Info compactă despre model
                total_params = sum(p.numel() for p in model.parameters())
                device = next(model.parameters()).device
                st.sidebar.caption(f"📊 {total_params:,} parametri | 🔧 {device}")

            else:
                st.sidebar.error(f"❌ {error}")
else:
    st.sidebar.error("❌ Nu s-au găsit modele")
    st.sidebar.caption("💡 Adaugă fișiere .pth în models/")

    # Buton pentru model fake pentru testare
    if st.sidebar.button("🎭 Model Simulat"):
        st.session_state.model = "fake_model"
        st.session_state.model_config = {'n_bands': 200, 'n_classes': 30}
        st.session_state.model_name = "Model Simulat"
        st.sidebar.success("✅ Model simulat!")

# 2. Load Image - compactat
st.sidebar.subheader("📂 Imagine")

# Încarcă lista imaginilor disponibile
available_images = load_available_images()

if available_images:
    selected_image = st.sidebar.selectbox(
        "Imagine:",
        available_images,
        help="Alege din setul de test UAV-HSI"
    )

    if st.sidebar.button("📂 Încarcă"):
        with st.spinner("Se încarcă..."):
            rs_data, gt_data = load_hyperspectral_image(selected_image)

            if rs_data is not None:
                st.session_state.rs_data = rs_data
                st.session_state.gt_data = gt_data
                st.session_state.image_name = selected_image

                # Folosește doar natural_colors din codul original
                st.session_state.rgb_composite = create_rgb_composite(rs_data)
                st.sidebar.success("✅ Încărcată!")
            else:
                st.sidebar.error("❌ Eroare!")

else:
    st.sidebar.error("❌ Nu s-au găsit imagini")
    st.sidebar.caption("Adaugă .npy în sample_data/rs/")

# 3. Butoane de acțiune - compacte
st.sidebar.subheader("🚀 Acțiuni")

# Buton pentru segmentare
segment_btn = st.sidebar.button("🚀 Segmentare", type="primary")

# Buton pentru indici vegetație
indices_btn = st.sidebar.button("🌿 Indici Vegetație")

# =============================================================================
# ZONA PRINCIPALĂ DE AFIȘARE
# =============================================================================

# Afișare imagine încărcată (RS + GT)
if 'rs_data' in st.session_state:

    # Info despre imagine
    st.info(f"📷 Imaginea încărcată: **{st.session_state.image_name}** | "
            f"Dimensiuni: {st.session_state.rs_data.shape}")

    # Afișare imagine originală și ground truth
    st.subheader("📷 Imagine și Ground Truth Interactive")

    # 3 coloane: RGB, GT, Legenda - AJUSTEZ PROPORȚIILE PENTRU MUTARE LA STÂNGA
    col1, col2, col3 = st.columns([0.9, 0.9, 1.0])  # Fac coloanele 1 și 2 mai mici pentru a împinge imaginile la stânga

    with col1:
        st.markdown("**🖼️ Compozit RGB**")
        if st.session_state.rgb_composite is not None:
            if PLOTLY_AVAILABLE:
                # FOLOSESC PLOTLY PENTRU RGB - ALIGNMENT PERFECT CU GT!
                fig_rgb = create_rgb_plotly_display(st.session_state.rgb_composite)
                if fig_rgb:
                    st.plotly_chart(fig_rgb, use_container_width=True,
                                    key="rgb_composit_chart",  # Cheie unică pentru RGB
                                    config={
                                        'displayModeBar': False,  # Ascund toolbar-ul
                                        'displaylogo': False,
                                        'staticPlot': True,  # RGB nu are nevoie de interactivitate
                                        'doubleClick': 'reset',
                                        'scrollZoom': False
                                    })
                else:
                    # Fallback la matplotlib dacă Plotly nu merge pentru RGB
                    fig, ax = plt.subplots(figsize=(4.8, 3.6), facecolor='none')
                    ax.imshow(st.session_state.rgb_composite, aspect='equal')
                    ax.axis('off')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
            else:
                # Fallback la matplotlib când Plotly nu e disponibil
                fig, ax = plt.subplots(figsize=(4.8, 3.6), facecolor='none')
                ax.imshow(st.session_state.rgb_composite, aspect='equal')
                ax.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with col2:
        st.markdown("**🎯 Ground Truth cu Hover Info**" if PLOTLY_AVAILABLE else "**🎯 Ground Truth cu Denumiri**")
        if st.session_state.gt_data is not None:
            if PLOTLY_AVAILABLE:
                # Folosește Plotly pentru interactivitate
                fig_gt = create_interactive_segmentation_plot(
                    st.session_state.gt_data,
                    ""  # Fără titlu redundant
                )
                st.plotly_chart(fig_gt, use_container_width=True,
                                key="gt_main",  # Cheie unică pentru GT principal
                                config={
                                    'displayModeBar': False,  # Ascund complet toolbar-ul pentru alignment perfect
                                    'displaylogo': False,  # Scoate logo-ul Plotly
                                    'staticPlot': False,  # Păstrez interactivitatea (hover)
                                    'doubleClick': 'reset',  # Double-click pentru reset
                                    'scrollZoom': False  # Dezactivez scroll zoom
                                })
            else:
                # Fallback la matplotlib
                fig = create_matplotlib_fallback(
                    st.session_state.gt_data,
                    "Ground Truth - Clase de Culturi"
                )
                st.pyplot(fig, use_container_width=True)
                plt.close()

        else:
            st.info("Nu există ground truth pentru această imagine")

    with col3:
        st.markdown("**📋 Legenda Claselor**")
        if st.session_state.gt_data is not None:
            # Găsește clasele prezente în imagine
            unique_classes = np.unique(st.session_state.gt_data)

            # Afișează legenda cu culori vizuale
            st.markdown("**Clase prezente:**")

            for class_id in sorted(unique_classes):
                if class_id < 30:  # Validare
                    class_name = UAV_HSI_CLASSES[class_id]["name_ro"]
                    color = UAV_HSI_CLASSES[class_id]["color"]

                    # Creează pătrat colorat HTML cu culoarea exactă
                    color_square = f'<span style="display:inline-block; width:20px; height:20px; background-color:{color}; border:1px solid #333; margin-right:8px; vertical-align:middle;"></span>'

                    # Afișează culoarea + numele clasei
                    st.markdown(f"{color_square} **{class_name}**", unsafe_allow_html=True)
                    st.markdown("")  # Spațiu între clase

            # Opțional: afișează legenda completă într-un expander
            with st.expander("🎨 Vezi toate clasele disponibile"):
                st.markdown("**Toate cele 30 de clase UAV-HSI:**")

                # Organizează în coloane pentru vizualizare mai bună
                cols = st.columns(2)
                for i in range(30):
                    class_info = UAV_HSI_CLASSES[i]
                    color_square = f'<span style="display:inline-block; width:15px; height:15px; background-color:{class_info["color"]}; border:1px solid #333; margin-right:5px; vertical-align:middle;"></span>'

                    # Alternează între coloane
                    with cols[i % 2]:
                        st.markdown(f"{color_square} **{i}:** {class_info['name_ro']}", unsafe_allow_html=True)

    # ==========================================================================
    # REZULTATE SEGMENTARE - APLICAREA MODELULUI REAL
    # ==========================================================================

    if segment_btn:
        if 'model' not in st.session_state:
            st.error("❌ Încarcă mai întâi un model!")
        else:
            st.session_state.segmentation_done = True

    if 'segmentation_done' in st.session_state and st.session_state.segmentation_done:
        st.subheader("🤖 Rezultate Segmentare" + (" Interactive" if PLOTLY_AVAILABLE else ""))

        if 'model' not in st.session_state:
            st.error("❌ Model nu este încărcat!")
        else:
            with st.spinner("🔄 Se aplică modelul de segmentare hiperspectrală..."):

                if st.session_state.model == "fake_model":
                    # Model simulat pentru testare
                    fake_prediction = np.random.randint(0, np.max(st.session_state.gt_data) + 1,
                                                        size=st.session_state.gt_data.shape)
                    st.session_state.prediction = fake_prediction
                    st.info("🎭 Folosind model simulat pentru demonstrație")

                else:
                    # Model real
                    try:
                        # Preprocessing - exact ca în codul tău
                        processed_image = preprocess_hyperspectral_image(st.session_state.rs_data)

                        # Verifică compatibilitatea benzilor
                        model_bands = st.session_state.model_config['n_bands']
                        image_bands = processed_image.shape[1]  # (1, C, H, W)

                        if image_bands != model_bands:
                            st.warning(f"⚠️ Nepotrivire benzi: Model={model_bands}, Imagine={image_bands}")

                            # Ajustează numărul de benzi
                            if image_bands > model_bands:
                                # Trunchiază la primele benzi
                                processed_image = processed_image[:, :model_bands, :, :]
                                st.info(f"📊 S-au folosit primele {model_bands} benzi din {image_bands}")
                            else:
                                # Repetă benzile dacă sunt mai puține
                                repeat_factor = model_bands // image_bands + 1
                                repeated = processed_image.repeat(1, repeat_factor, 1, 1)
                                processed_image = repeated[:, :model_bands, :, :]
                                st.info(f"📊 S-au repetat benzile pentru a ajunge la {model_bands}")

                        # Aplicarea modelului
                        prediction, error = apply_model_inference(st.session_state.model, processed_image)

                        if prediction is not None:
                            st.session_state.prediction = prediction
                            st.success(f"✅ Segmentare completă cu {st.session_state.model_name}!")

                            # Statistici predicție
                            unique_classes = np.unique(prediction)
                            total_pixels = prediction.size
                            st.info(f"📊 Predicție: {len(unique_classes)} clase detectate din {total_pixels:,} pixeli")

                        else:
                            st.error(f"❌ Eroare la aplicarea modelului: {error}")
                            # Fallback la model simulat
                            fake_prediction = np.random.randint(0, np.max(st.session_state.gt_data) + 1,
                                                                size=st.session_state.gt_data.shape)
                            st.session_state.prediction = fake_prediction
                            st.info("🎭 Folosind predicție simulată ca fallback")

                    except Exception as e:
                        st.error(f"❌ Eroare neașteptată: {str(e)}")
                        # Fallback la model simulat
                        fake_prediction = np.random.randint(0, np.max(st.session_state.gt_data) + 1,
                                                            size=st.session_state.gt_data.shape)
                        st.session_state.prediction = fake_prediction
                        st.info("🎭 Folosind predicție simulată ca fallback")

        # AFIȘARE PREDICȚIE VS GROUND TRUTH CU ALIGNMENT PERFECT
        if 'prediction' in st.session_state:
            col3, col4 = st.columns([0.9, 0.9])  # Coloane mai mici pentru a împinge imaginile la stânga

            with col3:
                st.markdown(
                    "**🔮 Predicția Modelului cu Hover Info**" if PLOTLY_AVAILABLE else "**🔮 Predicția Modelului**")

                model_name = st.session_state.get('model_name', 'Model Necunoscut')

                if PLOTLY_AVAILABLE:
                    # Folosește Plotly pentru interactivitate
                    fig_pred = create_interactive_segmentation_plot(
                        st.session_state.prediction,
                        ""  # Fără titlu redundant
                    )
                    st.plotly_chart(fig_pred, use_container_width=True,
                                    key="prediction_chart",  # Cheie unică pentru predicție
                                    config={
                                        'displayModeBar': False,  # Ascund complet toolbar-ul
                                        'displaylogo': False,  # Scoate logo-ul Plotly
                                        'staticPlot': False,  # Păstrez interactivitatea (hover)
                                        'doubleClick': 'reset',  # Double-click pentru reset
                                        'scrollZoom': False  # Dezactivez scroll zoom
                                    })
                else:
                    # Fallback la matplotlib
                    fig = create_matplotlib_fallback(
                        st.session_state.prediction,
                        f"Predicție - {model_name}"
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

            with col4:
                st.markdown(
                    "**🎯 Ground Truth cu Hover Info**" if PLOTLY_AVAILABLE else "**🎯 Ground Truth de Referință**")

                if PLOTLY_AVAILABLE:
                    # Plotly pentru ground truth în comparație
                    fig_gt_compare = create_interactive_segmentation_plot(
                        st.session_state.gt_data,
                        ""  # Fără titlu redundant
                    )
                    st.plotly_chart(fig_gt_compare, use_container_width=True,
                                    key="gt_comparison_chart",  # Cheie unică pentru GT comparație
                                    config={
                                        'displayModeBar': False,  # Ascund complet toolbar-ul
                                        'displaylogo': False,  # Scoate logo-ul Plotly
                                        'staticPlot': False,  # Păstrez interactivitatea (hover)
                                        'doubleClick': 'reset',  # Double-click pentru reset
                                        'scrollZoom': False  # Dezactivez scroll zoom
                                    })
                else:
                    # Fallback la matplotlib
                    fig = create_matplotlib_fallback(
                        st.session_state.gt_data,
                        "Ground Truth - Referință"
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

            # Metrici de performanță reale (dacă modelul real a fost aplicat)
            if 'model' in st.session_state and st.session_state.model != "fake_model":
                st.subheader("📊 Metrici de Performanță per Clasă")

                # Calculează metrici reale
                try:
                    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

                    # Flatten pentru calcule
                    gt_flat = st.session_state.gt_data.flatten()
                    pred_flat = st.session_state.prediction.flatten()

                    # Acuratețe
                    accuracy = accuracy_score(gt_flat, pred_flat)

                    # IoU pentru fiecare clasă
                    unique_classes = np.unique(gt_flat)
                    ious = []
                    class_performance = []

                    for cls in unique_classes[:10]:  # Primele 10 clase pentru afișare
                        if cls < 30:  # Validare
                            gt_mask = (gt_flat == cls)
                            pred_mask = (pred_flat == cls)
                            intersection = np.sum(gt_mask & pred_mask)
                            union = np.sum(gt_mask | pred_mask)
                            if union > 0:
                                iou = intersection / union
                                ious.append(iou)
                                class_performance.append({
                                    'Clasă': f"{int(cls)}: {UAV_HSI_CLASSES[cls]['name_ro']}",
                                    # FORȚEZ int() pentru cls
                                    'IoU': f"{iou:.3f}",
                                    'Pixeli GT': int(np.sum(gt_mask)),  # FORȚEZ int()
                                    'Pixeli Pred': int(np.sum(pred_mask))  # FORȚEZ int()
                                })

                    mean_iou = np.mean(ious) if ious else 0.0

                    # Afișează metrici generale
                    col5, col6, col7, col8 = st.columns(4)

                    with col5:
                        st.metric("Acuratețe Globală", f"{accuracy:.3f}")
                    with col6:
                        st.metric("IoU Mediu", f"{mean_iou:.3f}")
                    with col7:
                        st.metric("Clase Detectate", f"{len(np.unique(pred_flat))}")
                    with col8:
                        st.metric("Clase în GT", f"{len(unique_classes)}")

                    # Tabel cu performanțe pe clase
                    if class_performance:
                        st.markdown("**📋 Performanță pe tipuri de culturi (primele 10 clase):**")

                        # Afișează performanțele cu culori vizuale
                        for perf in class_performance:
                            class_id = int(perf['Clasă'].split(':')[0])
                            class_name = UAV_HSI_CLASSES[class_id]["name_ro"]
                            color = UAV_HSI_CLASSES[class_id]["color"]

                            # Pătrat colorat pentru fiecare clasă
                            color_square = f'<span style="display:inline-block; width:16px; height:16px; background-color:{color}; border:1px solid #333; margin-right:8px; vertical-align:middle;"></span>'

                            # Afișează performanța cu culoarea
                            col_name, col_iou, col_pixels = st.columns([3, 1, 2])

                            with col_name:
                                st.markdown(f"{color_square} **{class_name}**", unsafe_allow_html=True)
                            with col_iou:
                                st.metric("IoU", perf['IoU'])
                            with col_pixels:
                                st.write(f"GT: {perf['Pixeli GT']}, Pred: {perf['Pixeli Pred']}")

                            st.markdown("---")

                except ImportError:
                    st.info("💡 Instalează sklearn pentru metrici precise: pip install scikit-learn")
                except Exception as e:
                    st.warning(f"⚠️ Nu s-au putut calcula metricile: {str(e)}")

            else:
                # Metrici simulate pentru demonstrație
                st.subheader("📊 Metrici de Performanță (Simulate)")
                col5, col6, col7, col8 = st.columns(4)

                # Simulare metrici
                accuracy = np.random.uniform(0.75, 0.95)
                iou = np.random.uniform(0.65, 0.85)
                precision = np.random.uniform(0.70, 0.90)
                recall = np.random.uniform(0.68, 0.88)

                with col5:
                    st.metric("Acuratețe", f"{accuracy:.3f}")
                with col6:
                    st.metric("IoU Mediu", f"{iou:.3f}")
                with col7:
                    st.metric("Precizie", f"{precision:.3f}")
                with col8:
                    st.metric("Recall", f"{recall:.3f}")

    # ==========================================================================
    # INDICII DE VEGETAȚIE
    # ==========================================================================

    if indices_btn:
        st.session_state.indices_calculated = True

    if 'indices_calculated' in st.session_state and st.session_state.indices_calculated:
        st.subheader("🌿 Indici de Vegetație")

        with st.spinner("📊 Se calculează indicii de vegetație pentru dataset-ul UAV-HSI..."):
            ndvi = calculate_ndvi(st.session_state.rs_data)
            evi = calculate_evi(st.session_state.rs_data)
            st.session_state.ndvi = ndvi
            st.session_state.evi = evi

        # Afișare indici
        col9, col10, col11 = st.columns(3)

        with col9:
            st.markdown("**🌱 NDVI (Normalized Difference Vegetation Index)**")
            if st.session_state.ndvi is not None:
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(st.session_state.ndvi, cmap='RdYlGn', vmin=-1, vmax=1, aspect='equal')
                ax.set_title("NDVI - Indice Vegetație\n(Verde = Vegetație sănătoasă)", fontsize=12)
                ax.axis('off')

                plt.tight_layout()
                cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=20, pad=0.1)
                cbar.set_label('Valoare NDVI', fontsize=10)

                st.pyplot(fig, use_container_width=True)
                plt.close()

        with col10:
            st.markdown("**🌿 EVI (Enhanced Vegetation Index)**")
            if st.session_state.evi is not None:
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(st.session_state.evi, cmap='RdYlGn', vmin=-1, vmax=2, aspect='equal')
                ax.set_title("EVI - Indice Îmbunătățit\n(Corectat pentru sol)", fontsize=12)
                ax.axis('off')

                plt.tight_layout()
                cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=20, pad=0.1)
                cbar.set_label('Valoare EVI', fontsize=10)

                st.pyplot(fig, use_container_width=True)
                plt.close()

        with col11:
            # Statistici pentru indici
            st.markdown("**📊 Analiză Statistică**")

            if st.session_state.ndvi is not None:
                ndvi_mean = np.nanmean(st.session_state.ndvi)
                ndvi_std = np.nanstd(st.session_state.ndvi)

                st.write("**NDVI Statistics:**")
                st.write(f"• **Mediu:** {ndvi_mean:.3f}")
                st.write(f"• **Deviație std:** {ndvi_std:.3f}")
                st.write(f"• **Minim:** {np.nanmin(st.session_state.ndvi):.3f}")
                st.write(f"• **Maxim:** {np.nanmax(st.session_state.ndvi):.3f}")

                # Interpretare
                if ndvi_mean > 0.6:
                    st.success("🌱 Vegetație densă și sănătoasă")
                elif ndvi_mean > 0.3:
                    st.info("🌿 Vegetație moderată")
                else:
                    st.warning("🟫 Vegetație rară sau sol expus")

            if st.session_state.evi is not None:
                evi_mean = np.nanmean(st.session_state.evi)
                evi_std = np.nanstd(st.session_state.evi)

                st.write("**EVI Statistics:**")
                st.write(f"• **Mediu:** {evi_mean:.3f}")
                st.write(f"• **Deviație std:** {evi_std:.3f}")
                st.write(f"• **Minim:** {np.nanmin(st.session_state.evi):.3f}")
                st.write(f"• **Maxim:** {np.nanmax(st.session_state.evi):.3f}")

        # Analiză combinată NDVI-EVI
        if 'ndvi' in st.session_state and 'evi' in st.session_state:
            st.markdown("---")
            st.markdown("**🔬 Analiză Combinată NDVI-EVI pentru Caracterizarea Culturilor**")

            # Calculează corelația
            ndvi_flat = st.session_state.ndvi.flatten()
            evi_flat = st.session_state.evi.flatten()

            # Elimină NaN-urile
            valid_mask = ~(np.isnan(ndvi_flat) | np.isnan(evi_flat))
            ndvi_valid = ndvi_flat[valid_mask]
            evi_valid = evi_flat[valid_mask]

            if len(ndvi_valid) > 0:
                correlation = np.corrcoef(ndvi_valid, evi_valid)[0, 1]
                st.info(
                    f"📈 Corelația NDVI-EVI: {correlation:.3f} (valori apropiate de 1 indică consistență între indici)")

                # Interpretare pe tipuri de culturi identificate
                if 'gt_data' in st.session_state:
                    unique_classes = np.unique(st.session_state.gt_data)
                    vegetatie_classes = [2, 4, 5, 6, 7, 15, 22]  # Clase de vegetație verde

                    found_vegetation = [cls for cls in unique_classes if cls in vegetatie_classes and cls < 30]

                    if found_vegetation:
                        st.markdown("**🌱 Analiza vegetației identificate:**")

                        for cls in found_vegetation[:5]:  # Primele 5 clase
                            class_name = UAV_HSI_CLASSES[cls]["name_ro"]
                            color = UAV_HSI_CLASSES[cls]["color"]
                            color_square = f'<span style="display:inline-block; width:12px; height:12px; background-color:{color}; border:1px solid #333; margin-right:5px; vertical-align:middle;"></span>'

                            st.markdown(f"{color_square} **{class_name}** - Vegetație activă detectată",
                                        unsafe_allow_html=True)

else:
    # Stare inițială - nu e încărcată nicio imagine
    st.subheader("🚀 Începe prin a încărca o imagine hiperspectrală UAV-HSI")
    st.info(
        "👈 Folosește bara laterală pentru a selecta și încărca o imagine hiperspectrală din setul de test pentru segmentarea culturilor agricole.")

    # Afișare placeholder cu informații despre dataset
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Despre Dataset-ul UAV-HSI**")
        st.write("""
        - **200 benzi spectrale** (385-1024 nm)
        - **30 clase de culturi** agricole
        - Imagini captate cu **dronă UAV** 
        - Rezoluție spațială: **0.1m/pixel**
        - Zona de studiu: **Shenzhou, China**
        """)

    with col2:
        st.markdown("**🎯 Capabilități Demo**")
        st.write("""
        - **Segmentare semantică** cu deep learning
        - **Vizualizare optimizată** cu 30 culori distincte
        - **Indici de vegetație** (NDVI, EVI)
        - **Metrici de performanță** detaliiate
        - **Denumiri în română** pentru toate clasele
        """)

# Footer cu informații - compactat
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin: 5px 0;'>
🌱 Demo Segmentare UAV-HSI | Lucrare de Diplomă | 30 clase culturi, 200 benzi spectrale
</div>
""", unsafe_allow_html=True)