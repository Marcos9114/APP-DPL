import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
# from statsmodels.tsa.seasonal import seasonal_decompose
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import IsolationForest

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
    selected_alim = st.multiselect("Seleccione uno o más alimentadores (ALIM):", options=alim_unicos)

    if selected_alim:
        # Filtrar el DataFrame por los ALIM seleccionados
        filtered_df = df[df['ALIM'].isin(selected_alim)].sort_values(by='TIME')

        # Crear gráfico de línea con go.Figure
        fig = go.Figure()

        for alim in selected_alim:
            df_subset = filtered_df[filtered_df['ALIM'] == alim]
            fig.add_trace(go.Scatter(
                x=df_subset['TIME'], 
                y=df_subset['VALUE'], 
                mode='lines',
                name=alim
            ))

        # Personalización del gráfico
        fig.update_layout(
            title="Mediciones de Corriente por Alimentador (ALIM)",
            xaxis_title="Fecha y Hora",
            yaxis_title="Valor de Corriente",
            template='plotly_dark',
            legend_title="Alimentador",
            width=2000,  # Ajusta el ancho (en píxeles)
            height=600,  # Ajusta la altura (en píxeles)
            legend=dict(
            orientation="h",  # Cambia la orientación de la leyenda a horizontal
            y=-0.2,  # Mueve la leyenda debajo del gráfico
            x=0.5,  # Centra la leyenda
            xanchor="center"
            )
        )
        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=False)

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

            # Crear gráfico de línea con go.Figure
            fig_et = go.Figure()

            for trafo_label in selected_trafo_labels:
                df_subset_et = filtered_df_et[filtered_df_et['TRAFO_LABEL'] == trafo_label]
                fig_et.add_trace(go.Scatter(
                    x=df_subset_et['TIME'], 
                    y=df_subset_et['VALUE'], 
                    mode='lines',
                    name=trafo_label
                ))

            # Personalización del gráfico
            fig_et.update_layout(
                title="Mediciones de Potencia [kW] por Estación Transformadora y Transformador",
                xaxis_title="Fecha y Hora",
                yaxis_title="Valor de Potencia [kW]",
                template='plotly_dark',
                legend_title="Transformador",
                width=2000,  # Ajusta el ancho (en píxeles)
                height=600,  # Ajusta la altura (en píxeles)
                legend=dict(
                orientation="h",  # Cambia la orientación de la leyenda a horizontal
                y=-0.2,  # Mueve la leyenda debajo del gráfico
                x=0.5,  # Centra la leyenda
                xanchor="center"
                )
            )

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig_et, use_container_width=False)

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

    # Obtener los valores únicos de la columna NOMBRE
    lat_unicos = df_lat['NOMBRE'].unique()
    selected_lat = st.multiselect("Seleccione una o más Líneas de Alta Tensión (LAT):", options=lat_unicos)

    if selected_lat:
        # Filtrar el DataFrame por las LAT seleccionadas
        filtered_df_lat = df_lat[df_lat['NOMBRE'].isin(selected_lat)].sort_values(by='TIME')

        # Crear gráfico de línea con go.Figure
        fig_lat = go.Figure()

        for nombre in selected_lat:
            df_subset_lat = filtered_df_lat[filtered_df_lat['NOMBRE'] == nombre]
            fig_lat.add_trace(go.Scatter(
                x=df_subset_lat['TIME'], 
                y=df_subset_lat['VALUE'], 
                mode='lines',
                name=nombre
            ))

        # Personalización del gráfico
        fig_lat.update_layout(
            title="Mediciones de Corriente por Línea de Alta Tensión (LAT)",
            xaxis_title="Fecha y Hora",
            yaxis_title="Valor de Corriente",
            template='plotly_dark',
            legend_title="Línea de Alta Tensión",
            width=2000,  # Ajusta el ancho (en píxeles)
            height=600,  # Ajusta la altura (en píxeles)
            legend=dict(
            orientation="h",  # Cambia la orientación de la leyenda a horizontal
            y=-0.2,  # Mueve la leyenda debajo del gráfico
            x=0.5,  # Centra la leyenda
            xanchor="center"
            )
        )
        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig_lat, use_container_width=False)


    else:
        st.write("Por favor, seleccione al menos una línea de alta tensión (LAT).")

# Función para la página del Mapa de Reclamos
def page_mapa_reclamos():
    st.title("🗺️ Mapa de Reclamos")
    
    # Cargar datos del archivo Parquet
    ruta_archivo = r'C:\Users\marco\OneDrive\Escritorio\APPWEB-DPL\APP-DPL\Tablas\SANCIONES_30_31_x_y.parquet'
    df = cargar_datos(ruta_archivo)
    
    # Mostrar el dataframe como tabla para referencia
    st.write("Vista previa de los datos:", df.head())
    
    # Filtrar los datos para mostrar solo aquellos con sanción anual mayor a 0
    df_filtrado = df[df['SANCION_ANUAL'] > 0]
    
    # Crear un mapa interactivo usando Plotly
    fig = px.scatter_mapbox(
        df_filtrado, 
        lat="lat", 
        lon="lng", 
        hover_name="NOM_ALIM", 
        hover_data={"lat": False, "lng": False, "SANCION_ANUAL": True, "TARIFA": True}, 
        color="SANCION_ANUAL",
        color_continuous_scale=px.colors.sequential.Viridis,  # Puedes ajustar el color aquí
        title="Mapa de Reclamos",
        zoom=14,
        height=600
    )

    # Personalizar el mapa
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})

    # Mostrar el mapa en Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar el mapa en Streamlit
    st.plotly_chart(fig, use_container_width=True)


# def page_eda():
#     st.title("🔍 Análisis Exploratorio de Datos (EDA)")

#     # Cargar datos
#     ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
#     df = cargar_datos(ruta_archivo)

#     # Mostrar estadísticas descriptivas
#     st.subheader("Estadísticas Descriptivas")
#     st.write(df.describe())

#     # Distribución de valores
#     st.subheader("Distribución de Valores")
#     columna = st.selectbox("Seleccione una columna para visualizar su distribución:", df.columns)
#     fig = px.histogram(df, x=columna, title=f"Distribución de {columna}")
#     st.plotly_chart(fig)

#     # Detección de valores nulos
#     st.subheader("Valores Nulos")
#     st.write(df.isnull().sum())

# def page_series_tiempo():
#     st.title("⏳ Análisis de Series de Tiempo")

#     # Cargar datos
#     ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
#     df = cargar_datos(ruta_archivo)

#     # Seleccionar ALIM
#     alim_unicos = df['ALIM'].unique()
#     selected_alim = st.selectbox("Seleccione un alimentador (ALIM):", options=alim_unicos)

#     if selected_alim:
#         # Filtrar datos
#         filtered_df = df[df['ALIM'] == selected_alim].set_index('TIME')

#         # Descomposición de la serie de tiempo
#         st.subheader("Descomposición de la Serie de Tiempo")
#         decomposition = seasonal_decompose(filtered_df['VALUE'], model='additive', period=24)  # Ajusta el período según tus datos
#         st.write("Tendencia")
#         st.line_chart(decomposition.trend)
#         st.write("Estacionalidad")
#         st.line_chart(decomposition.seasonal)
#         st.write("Residuos")
#         st.line_chart(decomposition.resid)

# def page_modelado():
#     st.title("🤖 Modelado Predictivo")

#     # Cargar datos
#     ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
#     df = cargar_datos(ruta_archivo)

#     # Seleccionar ALIM
#     alim_unicos = df['ALIM'].unique()
#     selected_alim = st.selectbox("Seleccione un alimentador (ALIM):", options=alim_unicos)

#     if selected_alim:
#         # Filtrar datos
#         filtered_df = df[df['ALIM'] == selected_alim]

#         # Preparar datos para el modelo
#         filtered_df['TIME'] = pd.to_datetime(filtered_df['TIME'])
#         filtered_df['HORA'] = filtered_df['TIME'].dt.hour
#         filtered_df['DIA'] = filtered_df['TIME'].dt.day
#         filtered_df['MES'] = filtered_df['TIME'].dt.month

#         X = filtered_df[['HORA', 'DIA', 'MES']]
#         y = filtered_df['VALUE']

#         # Dividir datos en entrenamiento y prueba
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Entrenar modelo
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)

#         # Predecir
#         y_pred = model.predict(X_test)

#         # Mostrar métricas
#         st.subheader("Métricas del Modelo")
#         mse = mean_squared_error(y_test, y_pred)
#         st.write(f"Error Cuadrático Medio (MSE): {mse}")

#         # Gráfico de predicciones vs valores reales
#         st.subheader("Predicciones vs Valores Reales")
#         fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Valor Real', 'y': 'Predicción'}, title="Predicciones vs Valores Reales")
#         st.plotly_chart(fig)

# def page_modelado():
#     st.title("🤖 Modelado Predictivo")

#     # Cargar datos
#     ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
#     df = cargar_datos(ruta_archivo)

#     # Seleccionar ALIM
#     alim_unicos = df['ALIM'].unique()
#     selected_alim = st.selectbox("Seleccione un alimentador (ALIM):", options=alim_unicos)

#     if selected_alim:
#         # Filtrar datos
#         filtered_df = df[df['ALIM'] == selected_alim]

#         # Preparar datos para el modelo
#         filtered_df['TIME'] = pd.to_datetime(filtered_df['TIME'])
#         filtered_df['HORA'] = filtered_df['TIME'].dt.hour
#         filtered_df['DIA'] = filtered_df['TIME'].dt.day
#         filtered_df['MES'] = filtered_df['TIME'].dt.month

#         X = filtered_df[['HORA', 'DIA', 'MES']]
#         y = filtered_df['VALUE']

#         # Dividir datos en entrenamiento y prueba
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Entrenar modelo
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)

#         # Predecir
#         y_pred = model.predict(X_test)

#         # Mostrar métricas
#         st.subheader("Métricas del Modelo")
#         mse = mean_squared_error(y_test, y_pred)
#         st.write(f"Error Cuadrático Medio (MSE): {mse}")

#         # Gráfico de predicciones vs valores reales
#         st.subheader("Predicciones vs Valores Reales")
#         fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Valor Real', 'y': 'Predicción'}, title="Predicciones vs Valores Reales")
#         st.plotly_chart(fig)

# def page_anomalias():
#     st.title("🚨 Detección de Anomalías")

#     # Cargar datos
#     ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
#     df = cargar_datos(ruta_archivo)

#     # Seleccionar ALIM
#     alim_unicos = df['ALIM'].unique()
#     selected_alim = st.selectbox("Seleccione un alimentador (ALIM):", options=alim_unicos)

#     if selected_alim:
#         # Filtrar datos
#         filtered_df = df[df['ALIM'] == selected_alim]

#         # Entrenar modelo de detección de anomalías
#         model = IsolationForest(contamination=0.05, random_state=42)  # Ajusta el parámetro de contaminación
#         filtered_df['ANOMALIA'] = model.fit_predict(filtered_df[['VALUE']])

#         # Filtrar anomalías
#         anomalias = filtered_df[filtered_df['ANOMALIA'] == -1]

#         # Mostrar anomalías
#         st.subheader("Anomalías Detectadas")
#         st.write(anomalias)

#         # Gráfico de anomalías
#         st.subheader("Gráfico de Anomalías")
#         fig = px.scatter(filtered_df, x='TIME', y='VALUE', color='ANOMALIA', title="Detección de Anomalías")
#         st.plotly_chart(fig)


# Barra lateral para navegación
st.sidebar.title("Navegación")
pagina_seleccionada = st.sidebar.radio(
    "Seleccione la página",
    ["Bienvenida", "Corriente por Distribuidor", "Potencia por ET", "Corriente de LAT", "Mapa de Reclamos"] #"EDA", "Series de Tiempo", "Modelado", "Anomalías"
)

# Llamar a la página seleccionada
if pagina_seleccionada == "Bienvenida":
    page_bienvenida()
elif pagina_seleccionada == "Corriente por Distribuidor":
    page_corriente_dist()
elif pagina_seleccionada == "Potencia por ET":
    page_potencia_et()
elif pagina_seleccionada == "Corriente de LAT":
    page_corriente_lat()
# elif pagina_seleccionada == "EDA":
#     page_eda()
# elif pagina_seleccionada == "Series de Tiempo":
#     page_series_tiempo()
# elif pagina_seleccionada == "Modelado":
#     page_modelado()
# elif pagina_seleccionada == "Anomalías":
#     page_anomalias()
