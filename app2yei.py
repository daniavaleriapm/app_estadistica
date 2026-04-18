import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from google import genai

# 🎀 CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Stats & AI Pro", layout="wide")

# 🎨 PERSONALIZACIÓN (CSS)
st.markdown("""
<style>
    .stApp { background-color: #F4CAAB; }
    h1, h2, h3 { color: #4F766F; }
    .stButton>button {
        background-color: #E2988D;
        color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊✨ Estadística Interactiva con IA")

# ⚙️ CONFIGURACIÓN DE DATOS (SIDEBAR)
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.subheader("📂 Datos")
    source = st.radio("Fuente:", ["Sintéticos", "CSV"])

    if source == "Sintéticos":
        n_input = st.slider("Tamaño de muestra", 30, 500, 100)
        tipo_dist = st.selectbox("Distribución", ["Normal", "Sesgada", "Outliers"])
        if tipo_dist == "Normal":
            data = pd.Series(np.random.normal(50, 10, n_input))
        elif tipo_dist == "Sesgada":
            data = pd.Series(np.random.exponential(10, n_input))
        else:
            data = pd.Series(np.concatenate([np.random.normal(50, 5, n_input-5), [120, 130, 5, 2, 140]]))
    else:
        file = st.file_uploader("Sube CSV", type="csv")
        if file:
            df = pd.read_csv(file)
            col = st.selectbox("Variable", df.select_dtypes(include=np.number).columns)
            data = df[col].dropna()
        else:
            st.info("Sube un archivo para comenzar")
            st.stop()

# 📈 VISUALIZACIÓN
st.header("📈 Distribución de Datos")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(data, kde=True, color="#E2988D", ax=ax1)
    ax1.set_title("Histograma + KDE")
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=data, color="#4F766F", ax=ax2)
    ax2.set_title("Boxplot")
    st.pyplot(fig2)

# 📝 AUTOEVALUACIÓN
with st.expander("📝 Análisis del estudiante"):
    normal_check = st.radio("¿Distribución normal?", ["Sí", "No", "No sé"])
    sesgo_text = st.text_area("¿Hay sesgo u outliers?")

# 🧪 PRUEBA Z
st.header("🧪 Prueba Z")
n = len(data)
x_bar = data.mean()
colA, colB = st.columns(2)

with colA:
    h0 = st.number_input("H0: μ =", value=float(x_bar))
    sigma = st.number_input("σ poblacional", value=float(data.std()), min_value=0.1)
    alpha = st.slider("α", 0.01, 0.10, 0.05)
    tipo_test = st.selectbox("Tipo", ["Bilateral", "Derecha", "Izquierda"])
    if n < 30:
        st.warning("⚠️ n < 30, la prueba Z puede no ser válida")

z = (x_bar - h0) / (sigma / np.sqrt(n))
if tipo_test == "Bilateral":
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    zc = stats.norm.ppf(1 - alpha/2)
    reject = abs(z) > zc
elif tipo_test == "Derecha":
    p = 1 - stats.norm.cdf(z)
    zc = stats.norm.ppf(1 - alpha)
    reject = z > zc
else:
    p = stats.norm.cdf(z)
    zc = stats.norm.ppf(alpha)
    reject = z < zc

with colB:
    st.metric("Z", round(z, 4))
    st.metric("p-value", round(p, 4))
    st.subheader("Decisión:")
    if reject:
        st.error("🔴 Rechazar H0")
    else:
        st.success("🟢 No rechazar H0")

# 📉 GRÁFICA NORMAL
fig3, ax3 = plt.subplots()
x_plot = np.linspace(-4, 4, 1000)
y_plot = stats.norm.pdf(x_plot)
ax3.plot(x_plot, y_plot, color="#6D7172")
if tipo_test == "Bilateral":
    ax3.fill_between(x_plot, y_plot, where=(abs(x_plot) > zc), color="#E2988D", alpha=0.5)
elif tipo_test == "Derecha":
    ax3.fill_between(x_plot, y_plot, where=(x_plot > zc), color="#E2988D", alpha=0.5)
else:
    ax3.fill_between(x_plot, y_plot, where=(x_plot < zc), color="#E2988D", alpha=0.5)
ax3.axvline(z, color="#4F766F", linestyle="--")
st.pyplot(fig3)

# 🤖 ASISTENTE IA (BLOQUE CORREGIDO)
# 🤖 ASISTENTE IA (VERSIÓN MODERNA 2.0)
st.header("🤖 Asistente IA")

if api_key:
    decision_user = st.radio("Tu decisión basada en los resultados:", ["Rechazo H0", "No rechazo H0"])

    if st.button("Consultar IA"):
        try:
            # Creamos el cliente con la nueva librería
            client = genai.Client(api_key=api_key)
            
            # Usamos el modelo que apareció en tu lista como disponible
            model_id = "gemini-2.0-flash"

            prompt = f"""
            Actúa como un profesor de estadística. Analiza los resultados de esta prueba Z:
            - Media muestral: {x_bar:.2f}
            - Hipótesis Nula (H0): {h0}
            - Tamaño de muestra (n): {n}
            - Desviación estándar (sigma): {sigma}
            - Alpha (significancia): {alpha}
            - Tipo de prueba: {tipo_test}
            - Estadístico Z calculado: {z:.4f}
            - P-value: {p:.4f}

            El estudiante ha decidido: "{decision_user}".
            
            ¿Es correcta la decisión? Explica brevemente comparando el P-value con el Alpha. 
            Menciona si los supuestos de la prueba Z (n >= 30) se cumplen.
            """

            with st.spinner("La IA está analizando los datos con Gemini 2.0..."):
                # Nueva sintaxis para generar contenido
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt
                )
                
            st.info("### 📝 Análisis de la IA")
            st.write(response.text)

        except Exception as e:
            st.error(f"Hubo un problema técnico: {e}")
            st.info("Tip: Si el error persiste, verifica que instalaste la librería con 'pip install google-genai'")
else:
    st.warning("⚠️ Por favor, ingresa tu API Key en la barra lateral.")
