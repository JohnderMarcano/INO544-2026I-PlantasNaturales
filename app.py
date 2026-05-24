import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# 1. Configuración de página
st.set_page_config(page_title="Diagnóstico Botánico CNN | IUJO", page_icon="🌿", layout="wide")

# Inicializar estadísticas
if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'plantas': 0, 'objetos': 0}

# 2. Inyección de CSS Front-End
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .titulo-pro { text-align: center; color: #4ADE80; font-family: 'Segoe UI', sans-serif; font-weight: bold; margin-bottom: 0px; padding-top: 0.5rem;}
    .subtitulo-pro { text-align: center; color: #9CA3AF; font-size: 1rem; margin-bottom: 1.5rem; }
    div[data-testid="column"] {
        background-color: #1F2937;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stProgress > div > div > div > div { background-color: #4ADE80; } 
    div[data-testid="stMetricValue"] { color: white; }
    div[data-testid="stMetricDelta"] > div { color: #9CA3AF !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. Encabezado y KPIs
st.markdown("<h1 class='titulo-pro'> Sistema de Visión Artificial</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo-pro'>Diagnóstico Botánico Inmediato | Cátedra: Prof. David Hernández</p>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Análisis", st.session_state.stats['total'])
kpi2.metric("Plantas Naturales", st.session_state.stats['plantas'])
kpi3.metric("Objetos Detectados", st.session_state.stats['objetos'])
with kpi4:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(" Limpiar Historial", use_container_width=True):
        st.session_state.stats = {'total': 0, 'plantas': 0, 'objetos': 0}
        st.rerun()

st.markdown("---")

# Cargar el modelo ONNX
@st.cache_resource
def cargar_modelo():
    return ort.InferenceSession("model/modelo_plantas.onnx")

sesion = cargar_modelo()

# 4. Estructura Principal
col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown("###  Entrada de Matriz (224x224 RGB)")
    
    tab1, tab2 = st.tabs([" Cámara", " Archivo"])
    imagen_cargada = None
    
    with tab1:
        foto = st.camera_input("Captura instantánea")
        if foto: imagen_cargada = Image.open(foto)
            
    with tab2:
        archivo = st.file_uploader("Subir imagen", type=['jpg', 'jpeg', 'png'])
        if archivo: imagen_cargada = Image.open(archivo)

with col_der:
    st.markdown("###  Motor de Inferencia CNN")
    
    if imagen_cargada:
        with st.container(border=True):
            
            # Centrar y achicar la imagen usando 3 columnas invisibles
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(imagen_cargada, caption="Tensor Analizado", use_container_width=True)
            
            # Procesamiento Matemático Invisible
            img_lista = imagen_cargada.convert('RGB').resize((224, 224))
            img_array = np.array(img_lista).astype('float32')
            img_array = np.expand_dims(img_array, axis=0)
            
            input_name = sesion.get_inputs()[0].name
            output_name = sesion.get_outputs()[0].name
            resultado = sesion.run([output_name], {input_name: img_array})[0]
            probabilidad = float(resultado[0][0])
            
            st.divider()
            
            # --- VEREDICTO Y BARRA DE PROGRESO INMEDIATA ---
            if probabilidad > 0.5:
                confianza_planta = probabilidad * 100
                st.success("✅ **VERDICTO: PLANTA NATURAL**")
                st.progress(int(confianza_planta), text=f"Nivel de Certeza Botánica: {confianza_planta:.2f}%")
                
                # Para evitar doble conteo si se recarga la página
                if 'ultima_img' not in st.session_state or st.session_state.ultima_img != foto or archivo:
                     st.session_state.stats['plantas'] += 1
                     st.session_state.stats['total'] += 1
                     st.session_state.ultima_img = foto or archivo
                     
            else:
                confianza_objeto = (1 - probabilidad) * 100
                st.error("🛑 **VERDICTO: OBJETO / CONTROL**")
                st.progress(int(confianza_objeto), text=f"Nivel de Desviación Botánica: {confianza_objeto:.2f}%")
                
                if 'ultima_img' not in st.session_state or st.session_state.ultima_img != foto or archivo:
                     st.session_state.stats['objetos'] += 1
                     st.session_state.stats['total'] += 1
                     st.session_state.ultima_img = foto or archivo
                     
        with st.expander("⚙️ Ver matrices crudas del Tensor"):
            st.code(f"Shape: {img_array.shape}\nSigmoid binario: {probabilidad:.6f}")
                    
    else:
        st.info("ℹ️ Sistema en espera. Selecciona una fuente de entrada para comenzar el análisis instantáneo.")