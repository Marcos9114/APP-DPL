import pandas as pd
import streamlit as st
import plotly.express as px

# Función para cargar datos con caché
@st.cache_data
def cargar_datos(ruta):
    return pd.read_parquet(ruta)

# Función para la página de bienvenida
def page_bienvenida():
    st.title("👋 Bienvenido al Sistema de Monitoreo de Corriente")
    st.write("""
        Esta aplicación permite visualizar las mediciones de corriente:
        - **Corriente por Distribuidor**: Muestra las mediciones por alimentador.
        - **Potencia por ET**: Muestra las mediciones de potencia por Estación Transformadora.
        - **Corriente de LAT**: Muestra las mediciones de las líneas de Alta Tensión (LAT).
        
        Utilice la barra lateral para navegar entre las páginas.
    """)

# Función para la página de Corriente por Distribuidor
def page_corriente_dist():
    st.title("📊 Corriente por Distribuidor")

    # Cargar datos con caché
    ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
    df = cargar_datos(ruta_archivo)

    # Selección de ALIM
    alim_unicos = df['ALIM'].unique()
    selected_alim = st.multiselect(
        "Seleccione uno o más alimentadores (ALIM):",
        options=alim_unicos
    )

    if selected_alim:
        # Filtrar el DataFrame por los ALIM seleccionados
        filtered_df = df[df['ALIM'].isin(selected_alim)].sort_values(by='TIME')

        # Crear gráfico de línea con Plotly
        fig = px.line(
            filtered_df, x='TIME', y='VALUE', color='ALIM', 
            title="Mediciones de Corriente por Alimentador (ALIM)",
            labels={'TIME': 'Fecha y Hora', 'VALUE': 'Valor de Corriente'},
            template='plotly_dark'
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)
    else:
        st.write("Por favor, seleccione al menos un alimentador (ALIM).")

# Función para la página de Potencia por ET
def page_potencia_et():
    st.title("🏭 Potencia por Estación Transformadora (ET) y Transformador (TRAFO)")

    # Cargar datos con caché
    ruta_archivo_et = 'Tablas/potencia_ET_2024.parquet'
    df_et = cargar_datos(ruta_archivo_et)

    # Selección de uno o más ET
    et_unicos = df_et['ET'].unique()
    selected_et = st.multiselect("Seleccione una o más Estaciones Transformadoras (ET):", options=et_unicos)

    if selected_et:
        # Filtrar el DataFrame por las ET seleccionadas
        df_filtrado_et = df_et[df_et['ET'].isin(selected_et)]

        # Crear etiquetas combinadas para los transformadores
        df_filtrado_et['TRAFO_LABEL'] = df_filtrado_et['TRAFO'] + " (ET " + df_filtrado_et['ET'] + ")"
        
        # Selección de uno o más TRAFO con etiqueta combinada
        trafo_labels_unicos = df_filtrado_et['TRAFO_LABEL'].unique()
        selected_trafo_labels = st.multiselect("Seleccione uno o más Transformadores:", options=trafo_labels_unicos)

        if selected_trafo_labels:
            # Filtrar el DataFrame usando las etiquetas combinadas
            filtered_df_et = df_filtrado_et[df_filtrado_et['TRAFO_LABEL'].isin(selected_trafo_labels)].sort_values(by='TIME')

            # Crear gráfico de línea con Plotly
            fig_et = px.line(
                filtered_df_et, x='TIME', y='VALUE', color='TRAFO_LABEL',
                title="Mediciones de Potencia [kW] por Estación Transformadora y Transformador",
                labels={'TIME': 'Fecha y Hora', 'VALUE': 'Valor de Potencia [kW]'},
                template='plotly_dark'
            )

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig_et)
        else:
            st.write("Por favor, seleccione al menos un transformador (TRAFO).")
    else:
        st.write("Por favor, seleccione al menos una estación transformadora (ET).")

# Función para la página de Corriente de LAT (Líneas de Alta Tensión)
def page_corriente_lat():
    st.title("⚡ Corriente de Líneas de Alta Tensión (LAT)")

    # Cargar datos con caché
    ruta_archivo_lat = 'Tablas/corriente_LAT_2024.parquet'
    df_lat = cargar_datos(ruta_archivo_lat)

    # Verificar las columnas del DataFrame
    st.write("Columnas en el DataFrame de LAT:", df_lat.columns)

    # Obtener los valores únicos de la columna NOMBRE
    lat_unicos = df_lat['NOMBRE'].unique()
    selected_lat = st.multiselect("Seleccione una o más Líneas de Alta Tensión (LAT):", options=lat_unicos)

    if selected_lat:
        # Filtrar el DataFrame por las LAT seleccionadas
        filtered_df_lat = df_lat[df_lat['NOMBRE'].isin(selected_lat)].sort_values(by='TIME')

        # Crear gráfico de línea con Plotly
        fig_lat = px.line(
            filtered_df_lat, x='TIME', y='VALUE', color='NOMBRE', 
            title="Mediciones de Corriente por Línea de Alta Tensión (LAT)",
            labels={'TIME': 'Fecha y Hora', 'VALUE': 'Valor de Corriente'},
            template='plotly_dark'
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig_lat)
    else:
        st.write("Por favor, seleccione al menos una línea de alta tensión (LAT).")

# Barra lateral para navegación con radio
page = st.sidebar.radio("Seleccione la página", 
                        ["Bienvenida", "Corriente por Distribuidor", "Potencia por ET", "Corriente de LAT"])

# Llamar a la página seleccionada
if page == "Bienvenida":
    page_bienvenida()
elif page == "Corriente por Distribuidor":
    page_corriente_dist()
elif page == "Potencia por ET":
    page_potencia_et()
elif page == "Corriente de LAT":
    page_corriente_lat()

