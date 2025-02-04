import pandas as pd 
import streamlit as st
import plotly.express as px

# Ruta del archivo Parquet
ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'

# Leer el archivo Parquet con Pandas
df = pd.read_parquet(ruta_archivo)

# Mostrar las primeras filas del DataFrame para verificar la lectura
st.write("Vista previa de los datos:", df.head())

# Convertir la columna TIME al tipo datetime
df['TIME'] = pd.to_datetime(df['TIME'], format='%d/%m/%Y %H:%M:%S')

# Asegurar que la columna VALUE sea numérica
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

# Obtener la lista de valores únicos en la columna ALIM
alim_unicos = df['ALIM'].unique()

# Selección de ALIM en Streamlit
selected_alim = st.multiselect(
    "Seleccione uno o más alimentadores (ALIM):",
    options=alim_unicos
)

if selected_alim:
    # Filtrar el DataFrame por los ALIM seleccionados
    filtered_df = df[df['ALIM'].isin(selected_alim)].sort_values(by='TIME')

    # Crear gráfico de línea con Plotly
    fig = px.line(filtered_df, x='TIME', y='VALUE', color='ALIM', 
                  title="Mediciones de Corriente por Alimentador (ALIM)",
                  labels={'TIME': 'Fecha y Hora', 'VALUE': 'Valor de Corriente'},
                  template='plotly_dark')

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)
else:
    st.write("Por favor, seleccione al menos un alimentador (ALIM).")
