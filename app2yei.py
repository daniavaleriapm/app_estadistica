import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from openai import OpenAI
import json

def card_title(text):
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.65);
            border: 1px solid #EEDADF;
            border-radius: 18px;
            padding: 12px 18px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(110, 75, 70, 0.05);
        ">
            <h3 style="margin:0; color:#6E4B46;">{text}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------
st.set_page_config(page_title="Stats & AI Pro", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #FFF9F7;
        --panel: #FFF1F4;
        --card: #FFFFFF;
        --rose: #E8B7C8;
        --rose-dark: #D78FA8;
        --cream: #F8EDE7;
        --moka: #6E4B46;
        --brown: #8B5E57;
        --soft-line: #EEDADF;
        --text: #4B302D;
        --muted: #8A6F6A;
    }

    .stApp {
        background: linear-gradient(180deg, #FFF8FA 0%, #FFF6F2 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF1F4 0%, #FBEAE3 100%);
        border-right: 1px solid var(--soft-line);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: var(--moka) !important;
        letter-spacing: 0.3px;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.6rem;
    }

    h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-top: 1rem;
    }

    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    p, label, div, span {
        color: var(--text);
    }

    .stButton > button {
        background: linear-gradient(135deg, #E8B7C8 0%, #DFA0B6 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(215, 143, 168, 0.18);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #DFA0B6 0%, #D78FA8 100%);
        color: white;
        border: none;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--soft-line);
        border-radius: 18px;
        padding: 14px 12px;
        box-shadow: 0 6px 18px rgba(110, 75, 70, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--moka);
        font-family: 'Cormorant Garamond', serif;
        font-size: 2rem !important;
    }

    .stAlert {
        border-radius: 16px;
        border: 1px solid var(--soft-line);
    }

    .stExpander {
        border: 1px solid var(--soft-line) !important;
        border-radius: 18px !important;
        background: rgba(255,255,255,0.70);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {
        background-color: #FFFDFD !important;
        border-radius: 14px !important;
        border: 1px solid var(--soft-line) !important;
    }

    .stTextArea textarea {
        background-color: #FFFDFD !important;
        border-radius: 14px !important;
        border: 1px solid var(--soft-line) !important;
    }

    .stSlider [data-baseweb="slider"] {
        color: var(--rose-dark) !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    hr {
        border: none;
        border-top: 1px solid var(--soft-line);
    }
            
    /* 🔑 INPUTS (API KEY, number_input, text_input) */
input, textarea {
    color: #4B302D !important;
    -webkit-text-fill-color: #4B302D !important;
}

/* 🔑 Placeholder (texto gris tipo "escribe aquí") */
input::placeholder, textarea::placeholder {
    color: #8A6F6A !important;
    opacity: 1 !important;
}

/* 🔑 Labels (Hipótesis, sigma, etc.) */
label {
    color: #6E4B46 !important;
    font-weight: 600;
}

/* 🔑 Selectbox y number input texto */
div[data-baseweb="input"] input {
    color: #4B302D !important;
}

/* 🔑 TextArea (Describe lo que observas) */
textarea {
    color: #4B302D !important;
}

/* 🔑 Sidebar texto */
section[data-testid="stSidebar"] * {
    color: #4B302D !important;
}

/* 🔑 Radio buttons texto */
div[role="radiogroup"] label {
    color: #4B302D !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> ♡ Estadística Interactiva con IA ♡ </h1>", unsafe_allow_html=True)
st.caption("Un espacio delicado para explorar distribuciones, pruebas de hipótesis y apoyo inteligente.")

# ---------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------
def detectar_outliers_iqr(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    li = q1 - 1.5 * iqr
    ls = q3 + 1.5 * iqr
    outliers = series[(series < li) | (series > ls)]
    return outliers, li, ls

def evaluar_forma_distribucion(series: pd.Series):
    skewness = series.skew()
    kurtosis = series.kurt()
    outliers, _, _ = detectar_outliers_iqr(series)

    if abs(skewness) < 0.5:
        sesgo_desc = "La distribución parece aproximadamente simétrica."
    elif skewness > 0:
        sesgo_desc = "La distribución presenta sesgo positivo (cola hacia la derecha)."
    else:
        sesgo_desc = "La distribución presenta sesgo negativo (cola hacia la izquierda)."

    normalidad_aprox = abs(skewness) < 0.5 and len(outliers) <= max(2, int(0.05 * len(series)))

    return {
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "outliers_n": int(len(outliers)),
        "normalidad_aprox": normalidad_aprox,
        "descripcion": sesgo_desc
    }

def calcular_prueba_z(x_bar, mu0, sigma, n, alpha, tipo_test):
    z = (x_bar - mu0) / (sigma / np.sqrt(n))

    if tipo_test == "Bilateral":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        zc = stats.norm.ppf(1 - alpha / 2)
        reject = abs(z) > zc
        region_text = f"Rechazar H0 si Z < {-zc:.4f} o Z > {zc:.4f}"
    elif tipo_test == "Derecha":
        p = 1 - stats.norm.cdf(z)
        zc = stats.norm.ppf(1 - alpha)
        reject = z > zc
        region_text = f"Rechazar H0 si Z > {zc:.4f}"
    else:  # Izquierda
        p = stats.norm.cdf(z)
        zc = stats.norm.ppf(alpha)
        reject = z < zc
        region_text = f"Rechazar H0 si Z < {zc:.4f}"

    return z, p, zc, reject, region_text

def consultar_groq(api_key, resumen_dict):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    schema = {
        "type": "object",
        "properties": {
            "decision_correcta": {"type": "string"},
            "decision_estudiante_correcta": {"type": "string"},
            "explicacion_breve": {"type": "string"},
            "interpretacion_resultados": {"type": "string"},
            "revision_supuestos": {"type": "string"},
            "comentario_distribucion": {"type": "string"},
            "retroalimentacion_estudiante": {"type": "string"}
        },
        "required": [
            "decision_correcta",
            "decision_estudiante_correcta",
            "explicacion_breve",
            "interpretacion_resultados",
            "revision_supuestos",
            "comentario_distribucion",
            "retroalimentacion_estudiante"
        ],
        "additionalProperties": False
    }

    prompt = f"""
Eres un profesor de estadística inferencial.
Analiza únicamente el siguiente resumen estadístico de una muestra y una prueba Z.
No inventes datos. No uses datos crudos. Explica con lenguaje claro para un estudiante universitario.

Resumen:
{json.dumps(resumen_dict, ensure_ascii=False, indent=2)}

Tareas:
1. Indica si la decisión correcta es rechazar o no rechazar H0.
2. Evalúa si la decisión del estudiante coincide.
3. Explica brevemente comparando p-value con alpha y/o Z con región crítica.
4. Interpreta el resultado en contexto general.
5. Revisa si los supuestos de la prueba Z parecen razonables.
6. Comenta brevemente la forma de la distribución.
7. Da retroalimentación directa al estudiante.

Responde solo en JSON válido.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "Eres preciso, pedagógico y breve."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "retroalimentacion_estadistica",
                "schema": schema
            }
        }
    )

    content = response.choices[0].message.content
    return json.loads(content)

# ---------------------------------
# SIDEBAR
# ---------------------------------
with st.sidebar:
    st.header("● Configuración")
    api_key = st.text_input("Groq API Key", type="password")

    st.subheader("~ Datos")
    source = st.radio("Fuente:", ["Sintéticos", "CSV"])

    if source == "Sintéticos":
        n_input = st.slider("Tamaño de muestra", 30, 500, 100)
        tipo_dist = st.selectbox("Distribución", ["Normal", "Sesgada", "Outliers"])

        if tipo_dist == "Normal":
            data = pd.Series(np.random.normal(50, 10, n_input), name="Variable")
        elif tipo_dist == "Sesgada":
            data = pd.Series(np.random.exponential(10, n_input), name="Variable")
        else:
            base = np.random.normal(50, 5, n_input - 5)
            extremos = np.array([120, 130, 5, 2, 140])
            data = pd.Series(np.concatenate([base, extremos]), name="Variable")

    else:
        file = st.file_uploader("Sube CSV", type="csv")
        if file:
            df = pd.read_csv(file)
            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if not num_cols:
                st.error("El CSV no contiene columnas numéricas.")
                st.stop()

            col = st.selectbox("Variable", num_cols)
            data = df[col].dropna()
        else:
            st.info("Sube un archivo para comenzar.")
            st.stop()

# ---------------------------------
# RESUMEN DESCRIPTIVO
# ---------------------------------
n = len(data)
x_bar = data.mean()
s = data.std(ddof=1)
mediana = data.median()
minimo = data.min()
maximo = data.max()
forma = evaluar_forma_distribucion(data)

card_title("☆ Resumen descriptivo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("n", n)
c2.metric("Media", f"{x_bar:.4f}")
c3.metric("Mediana", f"{mediana:.4f}")
c4.metric("Desv. estándar muestral", f"{s:.4f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Mínimo", f"{minimo:.4f}")
c6.metric("Máximo", f"{maximo:.4f}")
c7.metric("Skewness", f"{forma['skewness']:.4f}")
c8.metric("Outliers (IQR)", forma["outliers_n"])

# ---------------------------------
# VISUALIZACIÓN
# ---------------------------------
card_title("《 Distribución de Datos 》")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig1.patch.set_facecolor("#FFF9F7")
    ax1.set_facecolor("#FFFFFF")

    ax1.hist(
        data,
        bins=20,
        density=True,
        alpha=0.75,
        color="#E8B7C8",
        edgecolor="#C98CA1",
        linewidth=1.0
    )

    xs = np.linspace(data.min(), data.max(), 300)
    kde = stats.gaussian_kde(data)
    ax1.plot(xs, kde(xs), color="#8B5E57", linewidth=2.2)

    ax1.set_title("Histograma + KDE", color="#6E4B46", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Valor", color="#4B302D")
    ax1.set_ylabel("Densidad", color="#4B302D")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#D9BFC6")
    ax1.spines["bottom"].set_color("#D9BFC6")
    ax1.tick_params(colors="#6E4B46")

    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(7,4))
    fig2.patch.set_facecolor("#FFF9F7")
    ax2.set_facecolor("#FFFFFF")

    ax2.boxplot(
        data,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#F8DDE6", color="#B67C8E", linewidth=1.5),
        medianprops=dict(color="#6E4B46", linewidth=2),
        whiskerprops=dict(color="#B67C8E", linewidth=1.5),
        capprops=dict(color="#B67C8E", linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor="#D78FA8", markeredgecolor="#D78FA8", alpha=0.7)
    )

    ax2.set_title("Boxplot", color="#6E4B46", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Valor", color="#4B302D")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#D9BFC6")
    ax2.spines["bottom"].set_color("#D9BFC6")
    ax2.tick_params(colors="#6E4B46")

    st.pyplot(fig2)

st.info(
    f"Evaluación automática preliminar: "
    f"{'Aproximadamente normal' if forma['normalidad_aprox'] else 'No claramente normal'} | "
    f"{forma['descripcion']} | "
    f"Outliers detectados: {forma['outliers_n']}"
)

# ---------------------------------
# AUTOEVALUACIÓN DEL ESTUDIANTE
# ---------------------------------
with st.expander("💐 Análisis del estudiante"):
    normal_check = st.radio("¿La distribución parece normal?", ["Sí", "No", "No sé"])
    sesgo_text = st.text_area("¿Hay sesgo u outliers? Describe lo que observas.")

# ---------------------------------
# PRUEBA Z
# ---------------------------------
card_title("🌸 Prueba Z para una media")

colA, colB = st.columns(2)

with colA:
    h0 = st.number_input("Hipótesis nula H0: μ =", value=float(x_bar))
    sigma = st.number_input("σ poblacional conocida", value=max(float(s), 0.1), min_value=0.1)
    alpha = st.slider("Nivel de significancia α", 0.01, 0.10, 0.05)
    tipo_test = st.selectbox("Tipo de prueba", ["Bilateral", "Derecha", "Izquierda"])

    st.markdown("*Hipótesis alternativa H1:*")
    if tipo_test == "Bilateral":
        st.write("H1: μ ≠ μ0")
    elif tipo_test == "Derecha":
        st.write("H1: μ > μ0")
    else:
        st.write("H1: μ < μ0")

    if n < 30:
        st.warning("⚠️ n < 30. La prueba Z puede no ser adecuada según lo pedido.")
    else:
        st.success(" 🪴n ≥ 30. Se cumple una de las condiciones pedidas para la prueba Z.")

z, p, zc, reject, region_text = calcular_prueba_z(x_bar, h0, sigma, n, alpha, tipo_test)

with colB:
    st.metric("Estadístico Z", f"{z:.4f}")
    st.metric("p-value", f"{p:.6f}")
    st.metric("Región crítica", region_text)

    st.subheader("Decisión automática")
    decision_automatica = "Rechazar H0" if reject else "No rechazar H0"

    if reject:
        st.error("🪷 Rechazar H0")
    else:
        st.success("🍃 No rechazar H0")

# ---------------------------------
# CURVA NORMAL CON REGIÓN CRÍTICA
# ---------------------------------
card_title("📉 Curva de decisión")

fig3, ax3 = plt.subplots(figsize=(9,4))
fig3.patch.set_facecolor("#FFF9F7")
ax3.set_facecolor("#FFFFFF")

x_plot = np.linspace(-4, 4, 1000)
y_plot = stats.norm.pdf(x_plot)

ax3.plot(x_plot, y_plot, color="#8B5E57", linewidth=2.3, label="N(0,1)")
ax3.axvline(z, color="#6E4B46", linestyle="--", linewidth=2, label=f"Z observado = {z:.4f}")

if tipo_test == "Bilateral":
    ax3.fill_between(x_plot, y_plot, where=(x_plot <= -zc), color="#E8B7C8", alpha=0.75)
    ax3.fill_between(x_plot, y_plot, where=(x_plot >= zc), color="#E8B7C8", alpha=0.75)
    ax3.axvline(-zc, color="#C98CA1", linestyle=":", linewidth=2)
    ax3.axvline(zc, color="#C98CA1", linestyle=":", linewidth=2)
elif tipo_test == "Derecha":
    ax3.fill_between(x_plot, y_plot, where=(x_plot >= zc), color="#E8B7C8", alpha=0.75)
    ax3.axvline(zc, color="#C98CA1", linestyle=":", linewidth=2)
else:
    ax3.fill_between(x_plot, y_plot, where=(x_plot <= zc), color="#E8B7C8", alpha=0.75)
    ax3.axvline(zc, color="#C98CA1", linestyle=":", linewidth=2)

ax3.set_title("Curva normal estándar con zona de rechazo", color="#6E4B46", fontsize=14, fontweight="bold")
ax3.set_xlabel("Z", color="#4B302D")
ax3.set_ylabel("Densidad", color="#4B302D")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_color("#D9BFC6")
ax3.spines["bottom"].set_color("#D9BFC6")
ax3.tick_params(colors="#6E4B46")
ax3.legend(frameon=False)
st.pyplot(fig3)

# ---------------------------------
# DECISIÓN DEL ESTUDIANTE
# ---------------------------------
card_title("☆ Comparación con el estudiante ☆")
decision_user = st.radio(
    "Tu decisión basada en los resultados:",
    ["Rechazo H0", "No rechazo H0"]
)

coincide = (
    (decision_user == "Rechazo H0" and decision_automatica == "Rechazar H0")
    or
    (decision_user == "No rechazo H0" and decision_automatica == "No rechazar H0")
)

if coincide:
    st.success("Tu decisión coincide con la decisión automática.")
else:
    st.warning("Tu decisión NO coincide con la decisión automática.")

# ---------------------------------
# MÓDULO DE IA CON GROQ
# ---------------------------------
card_title(" ●~ Asistente IA ~● ")

if api_key:
    if st.button("Consultar IA"):
        try:
            resumen_ia = {
                "variable": str(data.name) if data.name is not None else "Variable analizada",
                "n": int(n),
                "media_muestral": round(float(x_bar), 4),
                "mediana": round(float(mediana), 4),
                "desviacion_estandar_muestral": round(float(s), 4),
                "minimo": round(float(minimo), 4),
                "maximo": round(float(maximo), 4),
                "skewness": round(float(forma["skewness"]), 4),
                "outliers_iqr": int(forma["outliers_n"]),
                "normalidad_aproximada": bool(forma["normalidad_aprox"]),
                "descripcion_distribucion": forma["descripcion"],
                "hipotesis_nula": f"mu = {h0}",
                "hipotesis_alternativa": (
                    "mu ≠ mu0" if tipo_test == "Bilateral"
                    else "mu > mu0" if tipo_test == "Derecha"
                    else "mu < mu0"
                ),
                "tipo_prueba": tipo_test,
                "sigma_poblacional": round(float(sigma), 4),
                "alpha": round(float(alpha), 4),
                "estadistico_z": round(float(z), 4),
                "p_value": round(float(p), 6),
                "region_critica": region_text,
                "decision_automatica": decision_automatica,
                "decision_estudiante": decision_user,
                "cumple_n_mayor_igual_30": bool(n >= 30),
                "comentario_estudiante_distribucion": sesgo_text,
                "respuesta_estudiante_normalidad": normal_check
            }

            with st.spinner("Groq está analizando el resumen estadístico..."):
                respuesta_ia = consultar_groq(api_key, resumen_ia)

            st.subheader("📝 Respuesta de la IA")
            st.write(f"*Decisión correcta:* {respuesta_ia['decision_correcta']}")
            st.write(f"*¿La decisión del estudiante fue correcta?:* {respuesta_ia['decision_estudiante_correcta']}")
            st.write(f"*Explicación breve:* {respuesta_ia['explicacion_breve']}")
            st.write(f"*Interpretación:* {respuesta_ia['interpretacion_resultados']}")
            st.write(f"*Supuestos:* {respuesta_ia['revision_supuestos']}")
            st.write(f"*Distribución:* {respuesta_ia['comentario_distribucion']}")
            st.write(f"*Retroalimentación al estudiante:* {respuesta_ia['retroalimentacion_estudiante']}")

            st.download_button(
                "Descargar resumen JSON enviado a la IA",
                data=json.dumps(resumen_ia, ensure_ascii=False, indent=2),
                file_name="resumen_estadistico_ia.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Hubo un problema técnico con Groq: {e}")
            st.info("Verifica tu GROQ API Key y que tengas instaladas las dependencias correctas.")
else:
    st.warning("⚠️ Ingresa tu Groq API Key en la barra lateral.")
