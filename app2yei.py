#librerias utilizadas
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai

#configuracion de la pagina
st.set_page_config(page_title="Stats & AI Pro", layout="wide")

#personalizacion
st.markdown("""
<style>
    body {
        background-color: #F4CAAB;
    }
    .stApp {
        background-color: #F4CAAB;
    }
    h1, h2, h3 {
        color: #4F766F;
    }
    .stButton>button {
        background-color: #E2988D;
        color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

#titulo
st.title("📊✨ Estadística Interactiva con IA")

#configuracion de datos
with st.sidebar:
    st.header("⚙️ Configuración")

    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)

    st.subheader("📂 Datos")
    source = st.radio("Fuente:", ["Sintéticos", "CSV"])

    if source == "Sintéticos":
        n = st.slider("Tamaño de muestra", 30, 500, 100)
        tipo = st.selectbox("Distribución", ["Normal", "Sesgada", "Outliers"])

        if tipo == "Normal":
            data = pd.Series(np.random.normal(50, 10, n))
        elif tipo == "Sesgada":
            data = pd.Series(np.random.exponential(10, n))
        else:
            data = pd.Series(np.concatenate([
                np.random.normal(50, 5, n-5),
                [120,130,5,2,140]
            ]))

    else:
        file = st.file_uploader("Sube CSV", type="csv")
        if file:
            df = pd.read_csv(file)
            col = st.selectbox("Variable", df.select_dtypes(include=np.number).columns)
            data = df[col].dropna()
        else:
            st.stop()

#vizualizacion de graficas
st.header("📈 Distribución de Datos")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, color="#E2988D", ax=ax)
    ax.set_title("Histograma + KDE")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x=data, color="#4F766F", ax=ax)
    ax.set_title("Boxplot")
    st.pyplot(fig)

#autoevaluacion
    with st.expander("📝 Análisis del estudiante"):
    normal = st.radio("¿Distribución normal?", ["Sí", "No", "No sé"])
    sesgo = st.text_area("¿Hay sesgo u outliers?")

#prueba z y validacion
    st.header("🧪 Prueba Z")

n = len(data)
x_bar = data.mean()

colA, colB = st.columns(2)

with colA:
    h0 = st.number_input("H0: μ =", value=float(x_bar))
    sigma = st.number_input("σ poblacional", value=float(data.std()), min_value=0.1)
    alpha = st.slider("α", 0.01, 0.10, 0.05)
    tipo = st.selectbox("Tipo", ["Bilateral", "Derecha", "Izquierda"])

    if n < 30:
    st.warning("⚠️ n < 30, la prueba Z puede no ser válida")

#calculos
z = (x_bar - h0) / (sigma / np.sqrt(n))

if tipo == "Bilateral":
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    zc = stats.norm.ppf(1 - alpha/2)
    reject = abs(z) > zc

elif tipo == "Derecha":
    p = 1 - stats.norm.cdf(z)
    zc = stats.norm.ppf(1 - alpha)
    reject = z > zc

else:
    p = stats.norm.cdf(z)
    zc = stats.norm.ppf(alpha)
    reject = z < zc
