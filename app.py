import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import os
import cv2
import av

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =========================================================
# IMPORTAR ONNX
# =========================================================

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except:
    ONNX_AVAILABLE = False


# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Sistema de Visión Artificial",
    page_icon="🌿",
    layout="wide"
)


# =========================================================
# ESTILOS
# =========================================================

st.markdown("""
<style>

/* Fondo general */

.stApp {
    background:
        radial-gradient(circle at top left, rgba(0,255,120,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(0,255,120,0.08), transparent 22%),
        #111418;
    color: white;
}

/* Contenedor principal */

.block-container {
    padding-top: 2rem;
    max-width: 1350px;
}

/* Título */

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    color: white;
    margin-bottom: 0;
}

/* Subtítulo */

.sub-title {
    text-align: center;
    color: #d0d0d0;
    font-size: 15px;
    margin-bottom: 35px;
}

/* Tarjetas KPI */

.kpi-card {
    background: linear-gradient(135deg, #1d2128, #222831);
    border: 1px solid rgba(80,255,140,0.28);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 0 18px rgba(0,255,120,0.06);
}

/* Texto KPI */

.kpi-title {
    color: #cfcfcf;
    font-size: 14px;
}

/* Número KPI */

.kpi-value {
    color: white;
    font-size: 34px;
    font-weight: bold;
}

/* Panel principal */

.panel {
    position: relative;
    overflow: hidden;

    background: rgba(25,28,34,0.84);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 20px;

    padding: 22px;

    backdrop-filter: blur(10px);

    min-height: 560px;
}

/* Decoración hojas */

.panel::before {
    content: "🌿";
    position: absolute;
    right: 20px;
    bottom: 10px;
    font-size: 140px;
    opacity: 0.05;
}

.panel::after {
    content: "🍃";
    position: absolute;
    right: 120px;
    top: 20px;
    font-size: 90px;
    opacity: 0.04;
}

/* Título panel */

.panel-title {
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 20px;
}

/* Botón */

.stButton > button {

    width: 100%;
    height: 62px;

    border-radius: 14px;

    border: none;

    background: #dcdcdc;

    color: black;

    font-size: 16px;

    font-weight: 600;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {

    background: #1f232b;

    border-radius: 10px;

    padding: 10px 18px;

    color: white;
}

.stTabs [aria-selected="true"] {

    background: #84ff9d !important;

    color: black !important;
}

/* Upload */

.upload-box {

    border: 2px dashed rgba(255,255,255,0.12);

    border-radius: 18px;

    padding: 55px 20px;

    text-align: center;

    background: rgba(255,255,255,0.03);
}

/* Espera */

.wait-box {

    height: 380px;

    display: flex;

    align-items: center;

    justify-content: center;

    text-align: center;

    color: white;

    opacity: 0.9;

    font-size: 22px;
}

/* Línea */

hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# VARIABLES DE SESIÓN
# =========================================================

if "total" not in st.session_state:
    st.session_state.total = 0

if "plantas" not in st.session_state:
    st.session_state.plantas = 0

if "objetos" not in st.session_state:
    st.session_state.objetos = 0

if "hashes" not in st.session_state:
    st.session_state.hashes = set()


# =========================================================
# MODELO
# =========================================================

MODEL_PATH = "model/modelo_plantas.onnx"


@st.cache_resource
def cargar_modelo():

    if not ONNX_AVAILABLE:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return ort.InferenceSession(MODEL_PATH)

    except:
        return None


session = cargar_modelo()


# =========================================================
# PREPROCESAMIENTO
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    img_array = np.array(image).astype(np.float32)

    img_array = img_array / 255.0

    img_array = np.transpose(img_array, (2, 0, 1))

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =========================================================
# SIGMOIDE
# =========================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================================
# CÁMARA EN TIEMPO REAL
# =========================================================

class VideoProcessor(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)

        # ---------------------------------------------
        # DEMO
        # ---------------------------------------------

        if not ONNX_AVAILABLE or session is None:

            prob = 0.91
            es_planta = True

        # ---------------------------------------------
        # INFERENCIA REAL
        # ---------------------------------------------

        else:

            input_tensor = preprocess_image(pil_image)

            input_name = session.get_inputs()[0].name

            output_name = session.get_outputs()[0].name

            output = session.run(
                [output_name],
                {input_name: input_tensor}
            )

            raw_value = float(output[0][0][0])

            prob = sigmoid(raw_value)

            es_planta = prob > 0.5

        # ---------------------------------------------
        # TEXTO Y COLOR
        # ---------------------------------------------

        if es_planta:

            text = f"🌿 PLANTA NATURAL - {int(prob * 100)}%"

            color = (0, 255, 120)

        else:

            text = f"🛑 OBJETO - {int((1 - prob) * 100)}%"

            color = (0, 0, 255)

        # ---------------------------------------------
        # PANEL SUPERIOR
        # ---------------------------------------------

        cv2.rectangle(
            img,
            (15, 15),
            (470, 90),
            (15, 15, 15),
            -1
        )

        # ---------------------------------------------
        # TEXTO
        # ---------------------------------------------

        cv2.putText(
            img,
            text,
            (30, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            3
        )

        return img


# =========================================================
# ENCABEZADO
# =========================================================

st.markdown(
    "<div class='main-title'>🌿 Sistema de Visión Artificial</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Diagnóstico Botánico Inmediato | Cátedra: Prof. David Hernández</div>",
    unsafe_allow_html=True
)


# =========================================================
# MÉTRICAS
# =========================================================

k1, k2, k3, k4 = st.columns(4, gap="medium")

with k1:

    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Total Análisis</div>
        <div class='kpi-value'>{st.session_state.total}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:

    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Plantas Naturales</div>
        <div class='kpi-value'>{st.session_state.plantas}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:

    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Objetos Detectados</div>
        <div class='kpi-value'>{st.session_state.objetos}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:

    if st.button("🗑️ Limpiar Historial"):

        st.session_state.total = 0
        st.session_state.plantas = 0
        st.session_state.objetos = 0
        st.session_state.hashes = set()

        st.rerun()


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# COLUMNAS PRINCIPALES
# =========================================================

left_col, right_col = st.columns(2, gap="large")

uploaded_image = None
image_bytes = None


# =========================================================
# PANEL IZQUIERDO
# =========================================================

with left_col:

    st.markdown("<div class='panel'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='panel-title'>Entrada de Imagen (224x224 RGB)</div>",
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["📷 Cámara", "📁 Archivo"])

    # =====================================================
    # CÁMARA EN VIVO
    # =====================================================

    with tab1:

        webrtc_streamer(
            key="planta-camera",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            }
        )

    # =====================================================
    # SUBIR ARCHIVO
    # =====================================================

    with tab2:

        st.markdown(
            """
            <div class="upload-box">

                <h3 style="color:white; margin-bottom:10px;">
                    Sube o arrastra tu imagen de planta aquí
                </h3>

                <p style="color:#cfcfcf;">
                    Formatos: JPG, PNG | Max 200MB
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:

            uploaded_image = Image.open(uploaded_file)

            image_bytes = uploaded_file.getvalue()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PANEL DERECHO
# =========================================================

with right_col:

    st.markdown("<div class='panel'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='panel-title'>Panel de Diagnóstico CNN</div>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # ESPERA
    # -----------------------------------------------------

    if uploaded_image is None:

        st.markdown("""
        <div class='wait-box'>
            ⏳ Sistema listo. Usa la cámara en vivo o sube una imagen para iniciar el análisis.
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # RESULTADO IMAGEN
    # -----------------------------------------------------

    else:

        st.image(
            uploaded_image,
            width=320,
            caption="Imagen analizada"
        )

        st.markdown("---")

        # ---------------------------------------------
        # DEMO
        # ---------------------------------------------

        if not ONNX_AVAILABLE:

            prob = 0.91

            es_planta = True

            st.warning(
                "Modo demo activo. ONNX Runtime no es compatible con Python 3.14."
            )

        # ---------------------------------------------
        # ERROR MODELO
        # ---------------------------------------------

        elif session is None:

            st.error("No se pudo cargar el modelo ONNX.")

            st.stop()

        # ---------------------------------------------
        # INFERENCIA
        # ---------------------------------------------

        else:

            input_tensor = preprocess_image(uploaded_image)

            input_name = session.get_inputs()[0].name

            output_name = session.get_outputs()[0].name

            output = session.run(
                [output_name],
                {input_name: input_tensor}
            )

            raw_value = float(output[0][0][0])

            prob = sigmoid(raw_value)

            es_planta = prob > 0.5

        # ---------------------------------------------
        # EVITAR REPETIDOS
        # ---------------------------------------------

        image_hash = hashlib.md5(image_bytes).hexdigest()

        if image_hash not in st.session_state.hashes:

            st.session_state.hashes.add(image_hash)

            st.session_state.total += 1

            if es_planta:
                st.session_state.plantas += 1
            else:
                st.session_state.objetos += 1

        # ---------------------------------------------
        # RESULTADO VISUAL
        # ---------------------------------------------

        if es_planta:

            st.success("✅ VERDICTO: PLANTA NATURAL")

            st.progress(float(prob))

            st.write(f"Certeza: {int(prob * 100)}%")

        else:

            st.error("🛑 VERDICTO: OBJETO / CONTROL")

            st.progress(float(1 - prob))

            st.write(f"Desviación: {int((1 - prob) * 100)}%")

        # ---------------------------------------------
        # DETALLES
        # ---------------------------------------------

        with st.expander("⚙️ Ver detalles del tensor"):

            if ONNX_AVAILABLE and session is not None:

                st.write(f"Shape tensor: {input_tensor.shape}")

                st.write(f"Valor sigmoide: {prob:.6f}")

            else:

                st.write("Modo demo activo")

    st.markdown("</div>", unsafe_allow_html=True)