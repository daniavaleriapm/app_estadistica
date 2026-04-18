<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stats & AI Pro", layout="wide")
st.title("📊✨ Estadística Interactiva con IA")

with st.sidebar:
    st.header("Datos")
    source = st.radio("Fuente:", ["Sintéticos", "CSV"])

    if source == "Sintéticos":
        n = st.slider("Tamaño", 30, 500, 100)
        data = pd.Series(np.random.normal(50, 10, n))
    else:
        file = st.file_uploader("CSV", type="csv")
        if file:
            df = pd.read_csv(file)
            col = st.selectbox("Columna", df.columns)
            data = df[col].dropna()
        else:
            st.stop()
=======
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stats & AI Pro", layout="wide")
st.title("📊✨ Estadística Interactiva con IA")
>>>>>>> 7fa6fc08f07e46ceae76c38115f2091860d60531
