import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap
import statistics
from streamlit_folium import st_folium  # Importar st_folium

# Función para cargar datos con caché
@st.cache_data
def cargar_datos(ruta):
    return pd.read_parquet(ruta)

# Función para crear gráficos de línea
def crear_grafico_linea(df, x_col, y_col, nombre_col, titulo, x_titulo, y_titulo, leyenda_titulo):
    fig = go.Figure()
    for nombre in df[nombre_col].unique():
        df_subset = df[df[nombre_col] == nombre]
        fig.add_trace(go.Scatter(
            x=df_subset[x_col], 
            y=df_subset[y_col], 
            mode='lines',
            name=nombre
        ))
    fig.update_layout(
        title=titulo,
        xaxis_title=x_titulo,
        yaxis_title=y_titulo,
        template='plotly_dark',
        legend_title=leyenda_titulo,
        width=2000,
        height=600,
        legend=dict(
            orientation="h",
            y=-0.2,
            x=0.5,
            xanchor="center"
        )
    )
    return fig

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
    ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
    df = cargar_datos(ruta_archivo)
    alim_unicos = df['ALIM'].unique()
    selected_alim = st.multiselect("Seleccione uno o más alimentadores (ALIM):", options=alim_unicos)
    
    if selected_alim:
        filtered_df = df[df['ALIM'].isin(selected_alim)].sort_values(by='TIME')
        fig = crear_grafico_linea(filtered_df, 'TIME', 'VALUE', 'ALIM', 
                                  "Mediciones de Corriente por Alimentador (ALIM)", 
                                  "Fecha y Hora", "Valor de Corriente", "Alimentador")
        st.plotly_chart(fig, use_container_width=False)
    else:
        st.write("Por favor, seleccione al menos un alimentador (ALIM).")

# Función para la página de Potencia por ET
def page_potencia_et():
    st.title("🏭 Potencia por Estación Transformadora (ET) y Transformador (TRAFO)")
    ruta_archivo_et = 'Tablas/potencia_ET_2024.parquet'
    df_et = cargar_datos(ruta_archivo_et)
    et_unicos = df_et['ET'].unique()
    selected_et = st.multiselect("Seleccione una o más Estaciones Transformadoras (ET):", options=et_unicos)
    
    if selected_et:
        df_filtrado_et = df_et[df_et['ET'].isin(selected_et)]
        df_filtrado_et['TRAFO_LABEL'] = df_filtrado_et['TRAFO'] + " (ET " + df_filtrado_et['ET'] + ")"
        trafo_labels_unicos = df_filtrado_et['TRAFO_LABEL'].unique()
        selected_trafo_labels = st.multiselect("Seleccione uno o más Transformadores:", options=trafo_labels_unicos)
        
        if selected_trafo_labels:
            filtered_df_et = df_filtrado_et[df_filtrado_et['TRAFO_LABEL'].isin(selected_trafo_labels)].sort_values(by='TIME')
            fig_et = crear_grafico_linea(filtered_df_et, 'TIME', 'VALUE', 'TRAFO_LABEL', 
                                         "Mediciones de Potencia [kW] por Estación Transformadora y Transformador", 
                                         "Fecha y Hora", "Valor de Potencia [kW]", "Transformador")
            st.plotly_chart(fig_et, use_container_width=False)
        else:
            st.write("Por favor, seleccione al menos un transformador (TRAFO).")
    else:
        st.write("Por favor, seleccione al menos una estación transformadora (ET).")

# Función para la página de Corriente de LAT (Líneas de Alta Tensión)
def page_corriente_lat():
    st.title("⚡ Corriente de Líneas de Alta Tensión (LAT)")
    ruta_archivo_lat = 'Tablas/corriente_LAT_2024.parquet'
    df_lat = cargar_datos(ruta_archivo_lat)
    lat_unicos = df_lat['NOMBRE'].unique()
    selected_lat = st.multiselect("Seleccione una o más Líneas de Alta Tensión (LAT):", options=lat_unicos)
    
    if selected_lat:
        filtered_df_lat = df_lat[df_lat['NOMBRE'].isin(selected_lat)].sort_values(by='TIME')
        fig_lat = crear_grafico_linea(filtered_df_lat, 'TIME', 'VALUE', 'NOMBRE', 
                                      "Mediciones de Corriente por Línea de Alta Tensión (LAT)", 
                                      "Fecha y Hora", "Valor de Corriente", "Línea de Alta Tensión")
        st.plotly_chart(fig_lat, use_container_width=False)
    else:
        st.write("Por favor, seleccione al menos una línea de alta tensión (LAT).")

# Función para la página del Mapa de Reclamos
def page_mapa_reclamos():
    st.title("🗺️ Mapa de Reclamos")
    ruta_archivo = 'Tablas/SANCIONES_30_31_x_y.parquet'
    df = cargar_datos(ruta_archivo)
    
    # Filtrar datos para incluir solo aquellos con SANCION_ANUAL > 0
    df_filtrado = df[df['SANCION_ANUAL'] > 0]
    
    # Extraer las coordenadas y el monto de SANCION_ANUAL
    lats = df_filtrado['lat'].tolist()
    longs = df_filtrado['lng'].tolist()
    sanciones = df_filtrado['SANCION_ANUAL'].tolist()
    
    # Calcular los valores medios de latitud y longitud
    meanLat = statistics.mean(lats)
    meanLong = statistics.mean(longs)
    
    # Crear un objeto mapa base usando Map()
    mapObj = folium.Map(location=[meanLat, meanLong], zoom_start=14.5)
    
    # Crear capa de mapa de calor (basado en SANCION_ANUAL)
    heatmap_sanciones = HeatMap(
        list(zip(lats, longs, sanciones)),  # Combinar lat, lng y SANCION_ANUAL
        min_opacity=0.5,                    # Opacidad mínima
        max_val=max(sanciones),             # Valor máximo para la escala de colores
        radius=25,                          # Radio de cada punto
        blur=15,                            # Desenfoque
        max_zoom=1                          # Nivel máximo de zoom
    )
    
    # Añadir capa de mapa de calor al mapa base
    heatmap_sanciones.add_to(mapObj)
    
    # Mostrar el primer mapa en Streamlit usando st_folium
    st.write("### Mapa de Calor: Intensidad basada en el monto de SANCION_ANUAL")
    st_folium(mapObj, width=700, height=600)
    
    # Segundo mapa de calor: Cantidad de reclamos por ubicación
    st.write("### Mapa de Calor: Intensidad basada en la cantidad de reclamos por ubicación")
    
    # Contar la cantidad de reclamos por ubicación
    from collections import defaultdict
    ubicaciones = defaultdict(int)
    for lat, lng in zip(lats, longs):
        ubicaciones[(lat, lng)] += 1
    
    # Preparar los datos para el mapa de calor
    heat_data_cantidad = [[lat, lng, count] for (lat, lng), count in ubicaciones.items()]
    
    # Crear un segundo objeto mapa base
    mapObj_cantidad = folium.Map(location=[meanLat, meanLong], zoom_start=14.5)
    
    # Crear capa de mapa de calor (basado en la cantidad de reclamos)
    heatmap_cantidad = HeatMap(
        heat_data_cantidad,                 # Combinar lat, lng y cantidad de reclamos
        min_opacity=0.5,                    # Opacidad mínima
        max_val=max([count for _, _, count in heat_data_cantidad]),  # Valor máximo para la escala de colores
        radius=25,                          # Radio de cada punto
        blur=15,                            # Desenfoque
        max_zoom=1                          # Nivel máximo de zoom
    )
    
    # Añadir capa de mapa de calor al segundo mapa base
    heatmap_cantidad.add_to(mapObj_cantidad)
    
    # Mostrar el segundo mapa en Streamlit usando st_folium
    st_folium(mapObj_cantidad, width=700, height=600)

# Nota: Para usar st_folium, necesitas instalar la biblioteca `streamlit-folium`:
# pip install streamlit-folium


# Barra lateral para navegación
st.sidebar.title("Navegación")
pagina_seleccionada = st.sidebar.radio(
    "Seleccione la página",
    ["Bienvenida", "Corriente por Distribuidor", "Potencia por ET", "Corriente de LAT", "Mapa de Reclamos"] 
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
elif pagina_seleccionada == "Mapa de Reclamos":
    page_mapa_reclamos()