import pandas as pd
import streamlit as st
import plotly.express as px

# Ruta del archivo CSV
ruta_archivo = 'Tablas/corrientes_alimentador_NOV2024.csv'

# Leer el archivo CSV con Pandas
df = pd.read_csv(ruta_archivo, delimiter=',')

# Mostrar las primeras filas del DataFrame para verificar la lectura
st.write("Vista previa de los datos:", df.head())

# Convertir la columna TIME al tipo datetime
df['TIME'] = pd.to_datetime(df['TIME'], format='%d/%m/%Y %H:%M:%S')

# Convertir la columna VALUE a numérica (manejar comas como separadores decimales)
df['VALUE'] = df['VALUE'].str.replace(',', '.').astype(float)

# Obtener la lista de valores únicos en la columna ALIM
alim_unicos = df['ALIM'].unique()

# Selección de ALIM en Streamlit
selected_alim = st.multiselect(
    "Seleccione uno o más alimentadores (ALIM):",
    options=alim_unicos
)

if selected_alim:
    # Filtrar el DataFrame por los ALIM seleccionados
    filtered_df = df[df['ALIM'].isin(selected_alim)]

    # Ordenar el DataFrame por la columna TIME
    filtered_df = filtered_df.sort_values(by='TIME', ascending=True)

    # Crear gráfico de línea con Plotly
    fig = px.line(filtered_df, x='TIME', y='VALUE', color='ALIM', 
                  title="Mediciones de Corriente por Alimentador (ALIM)",
                  labels={'TIME': 'Fecha y Hora', 'VALUE': 'Valor de Corriente'},
                  template='plotly_dark')

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

else:
    st.write("Por favor, seleccione al menos un alimentador (ALIM).")