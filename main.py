import os
import dask.dataframe as dd
import pandas as pd
import streamlit as st
import plotly.express as px

# Ruta de la carpeta donde están los archivos
carpeta = './medicionesI'

# Listar todos los archivos en la carpeta
archivos = [f for f in os.listdir(carpeta) if f.endswith('.txt')]


@st.cache_data
def leer_y_combinar_archivos_dask():
    # Leer y combinar todos los archivos con Dask
    dataframes = []
    for idx, archivo in enumerate(archivos):
        ruta_archivo = os.path.join(carpeta, archivo)

        # Leer el archivo asegurando que el encabezado solo se lea una vez
        if idx == 0:
            # El primer archivo puede tener encabezado
            df = dd.read_csv(ruta_archivo, delimiter=';', header=0)
        else:
            # Los siguientes archivos no deben tener encabezado
            df = dd.read_csv(ruta_archivo, delimiter=';', header=None)
            df.columns = ['NAME', 'TIME', 'VALUE']

        dataframes.append(df)

    # Combinar todos los DataFrames
    df_combined = dd.concat(dataframes, axis=0)

    return df_combined


@st.cache_data
def cargar_distribuidores():
    # Cargar el archivo de distribuidores
    ruta_distribuidores = './Tablas/Listado_de_distribuidores_Hoja1.csv'

    # Leer el archivo con Dask
    df_distribuidores = dd.read_csv(ruta_distribuidores, delimiter=',', header=0)
    df_distribuidores.columns = [
        'ET', 'ALIM', 'BARRA', 'TRAFO', 'SCADA'
    ]  # Aseguramos que las columnas estén bien definidas

    # Limpiar nombres de columnas
    df_distribuidores.columns = df_distribuidores.columns.str.strip()

    return df_distribuidores


# Cargar datos con Dask
df_combined = leer_y_combinar_archivos_dask()

# Eliminar las filas donde la columna 'TIME' contenga el texto "TIME"
df_combined = df_combined[df_combined['TIME'] != 'TIME']

# Convertir la columna TIME al tipo datetime
df_combined['TIME'] = dd.to_datetime(df_combined['TIME'], format='%d/%m/%Y %H:%M:%S')

# Cargar distribuidores
codigos_distribuidores = cargar_distribuidores()

# Selección de distribuidores en Streamlit
selected_distribuidores = st.multiselect(
    "Distribuidores disponibles:",
    options=codigos_distribuidores['ALIM'].compute()
)

if selected_distribuidores:
    # Filtrar códigos de distribuidores seleccionados
    filtered_distribuidores = codigos_distribuidores[
        codigos_distribuidores['ALIM'].isin(selected_distribuidores)
    ].compute()

    # Realizar el merge entre corrientes_SCADA y codigos_distribuidores
    result_df = dd.merge(
        df_combined,
        filtered_distribuidores,
        left_on='NAME',
        right_on='SCADA'
    ).compute()

    # Seleccionar columnas de interés
    result_df = result_df[['TIME', 'VALUE', 'ET', 'ALIM']]

    # Convertir la columna VALUE a numérica
    result_df['VALUE'] = pd.to_numeric(result_df['VALUE'], errors='coerce')
    result_df = result_df.sort_values(by='TIME', ascending=True)

    # Validar y convertir columnas al formato esperado
    result_df['TIME'] = pd.to_datetime(result_df['TIME'], errors='coerce')
    result_df['VALUE'] = result_df['VALUE'].astype(float)

    # Crear gráfico de línea con Plotly
    fig = px.line(result_df, x='TIME', y='VALUE', color='ALIM', 
                  title="Mediciones de Corriente con Línea",
                  labels={'TIME': 'Fecha y Hora', 'VALUE': 'Valor de Corriente'},
                  template='plotly_dark')

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

else:
    st.write("Por favor, seleccione al menos un distribuidor.")
