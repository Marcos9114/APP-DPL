import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap
import statistics
from streamlit_folium import st_folium
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime

# Función para cargar datos con caché
@st.cache_data
def cargar_datos(ruta):
    return pd.read_parquet(ruta)

@st.cache_data
def cargar_datos_factibilidades(ruta_archivo):
    """Carga los datos desde el archivo Excel y devuelve un DataFrame."""
    return pd.read_excel(ruta_archivo)

def calcular_color(valor, max_valor):
    escala = int(255 * (valor / max_valor))
    return f"rgb({255 - escala}, {escala}, 100)"

def format_date(dt):
    return f"{dt.day}/{dt.month}/{dt.year}"

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
    
    # Botón para mostrar temperatura
    mostrar_temperatura = st.button("Ver Temperatura")
    
    # Botón para mostrar T° max diaria vs I max diaria
    mostrar_temp_vs_corriente = st.button("T° max diaria vs I max diaria")
    
    if selected_alim:
        filtered_df = df[df['ALIM'].isin(selected_alim)].sort_values(by='TIME')
        
        # Crear gráfico de corriente
        fig_corriente = crear_grafico_linea(filtered_df, 'TIME', 'VALUE', 'ALIM', 
                                  "Mediciones de Corriente por Alimentador (ALIM)", 
                                  "Fecha y Hora", "Valor de Corriente", "Alimentador")
        
        # Mostrar el gráfico de corriente
        st.plotly_chart(fig_corriente, use_container_width=False)
        
        # Si se activa el botón "Ver Temperatura", cargar y mostrar datos de temperatura
        if mostrar_temperatura:
            ruta_temperatura = 'Tablas/mendoza_aero.parquet'
            df_temperatura = cargar_datos(ruta_temperatura)
            
            # Asegurarse de que la columna FECHA esté en formato datetime
            df_temperatura['FECHA'] = pd.to_datetime(df_temperatura['FECHA'])
            
            # Ordenar los datos de temperatura por fecha
            df_temperatura = df_temperatura.sort_values(by='FECHA')
            
            # Verificar y limpiar los datos de temperatura
            df_temperatura['TEMP'] = pd.to_numeric(df_temperatura['TEMP'], errors='coerce')
            df_temperatura = df_temperatura.dropna(subset=['TEMP'])
            
            # Crear gráfico de temperatura
            fig_temperatura = go.Figure()
            fig_temperatura.add_trace(go.Scatter(
                x=df_temperatura['FECHA'], 
                y=df_temperatura['TEMP'], 
                mode='lines',
                name='Temperatura (Mendoza Aero)'
            ))
            fig_temperatura.update_layout(
                title='Temperatura Registrada',
                xaxis_title='Fecha y Hora',
                yaxis_title='Temperatura (°C)',
                template='plotly_dark',
                width=2000,
                height=600,
                yaxis=dict(
                    range=[df_temperatura['TEMP'].min() - 5, df_temperatura['TEMP'].max() + 5]  # Ajustar el rango del eje Y
                )
            )
            
            # Mostrar el gráfico de temperatura
            st.plotly_chart(fig_temperatura, use_container_width=False)
        
        # Si se activa el botón "T° max diaria vs I max diaria", crear y mostrar el gráfico correspondiente
        if mostrar_temp_vs_corriente:
            ruta_temperatura = 'Tablas/mendoza_aero.parquet'
            df_temperatura = cargar_datos(ruta_temperatura)
            
            # Asegurarse de que la columna FECHA esté en formato datetime
            df_temperatura['FECHA'] = pd.to_datetime(df_temperatura['FECHA'])
            
            # Calcular la temperatura máxima diaria
            df_temperatura['FECHA'] = df_temperatura['FECHA'].dt.date
            temp_max_diaria = df_temperatura.groupby('FECHA')['TEMP'].max().reset_index()
            
            # Calcular la corriente máxima diaria
            filtered_df['FECHA'] = pd.to_datetime(filtered_df['TIME']).dt.date
            corriente_max_diaria = filtered_df.groupby('FECHA')['VALUE'].max().reset_index()
            
            # Crear gráfico de T° max diaria vs I max diaria
            fig_temp_vs_corriente = go.Figure()
            fig_temp_vs_corriente.add_trace(go.Scatter(
                x=corriente_max_diaria['FECHA'], 
                y=corriente_max_diaria['VALUE'], 
                mode='lines',
                name='Corriente Máxima Diaria',
                line=dict(color='lightblue')
            ))
            fig_temp_vs_corriente.add_trace(go.Scatter(
                x=temp_max_diaria['FECHA'], 
                y=temp_max_diaria['TEMP'], 
                mode='lines',
                name='Temperatura Máxima Diaria',
                line=dict(color='orange')
            ))
            fig_temp_vs_corriente.update_layout(
                title='Temperatura Máxima Diaria vs Corriente Máxima Diaria',
                xaxis_title='Fecha',
                yaxis_title='Valor',
                template='plotly_dark',
                width=2000,
                height=600,
                legend=dict(
                    orientation="h",
                    y=-0.2,
                    x=0.5,
                    xanchor="center"
                )
            )
            
            # Mostrar el gráfico
            st.plotly_chart(fig_temp_vs_corriente, use_container_width=False)
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
    st.title("🗺️ Mapas de Sanciones por Distribuidor")

    # Cargar datos
    ruta_archivo = 'Tablas/SANCIONES_30_31_x_y.parquet'
    df = cargar_datos(ruta_archivo)

    st.write("### Selección de visualización:")
    
    # Selectbox para elegir solo un mapa a la vez
    opciones_mapa = ["Mapa de Calor en función del Volumen de Reclamos", 
                     "Mapa de Calor en función del Costo de Sanción", 
                     "Visualización de Clientes Reclamantes"]
    
    mapa_seleccionado = st.selectbox("Seleccione el mapa que desea visualizar:", opciones_mapa)

    # Multiselect para seleccionar distribuidores
    st.write("### Selección de Distribuidores:")
    distribuidor_unicos = sorted(df['NOM_ALIM'].unique())
    selected_distribuidores = st.multiselect("Seleccione uno o más distribuidores:", options=distribuidor_unicos)

    if selected_distribuidores:
        # Filtrar datos por los distribuidores seleccionados
        df_filtrado = df[df['NOM_ALIM'].isin(selected_distribuidores) & (df['SANCION_ANUAL'] > 0)]
        lats = df_filtrado['lat'].tolist()
        longs = df_filtrado['lng'].tolist()
        sanciones = df_filtrado['SANCION_ANUAL'].tolist()

        meanLat = statistics.mean(lats) if lats else -34.6037
        meanLong = statistics.mean(longs) if longs else -58.3816

        # Mostrar el mapa seleccionado
        if mapa_seleccionado == "Mapa de Calor en función del Volumen de Reclamos":
            st.write("### Mapa de Calor en función del Volumen de Reclamos")
            ubicaciones = defaultdict(int)
            for lat, lng in zip(lats, longs):
                ubicaciones[(lat, lng)] += 1
            heat_data = [[lat, lng, count] for (lat, lng), count in ubicaciones.items()]

            mapObj_calor = folium.Map(location=[meanLat, meanLong], zoom_start=14)
            HeatMap(heat_data, radius=15, blur=20, max_zoom=1).add_to(mapObj_calor)
            st_folium(mapObj_calor, width=700, height=500)

        elif mapa_seleccionado == "Mapa de Calor en función del Costo de Sanción":
            st.write("### Mapa de Calor en función del Costo de Sanción")
            heat_data_costo = [[lat, lng, sancion] for lat, lng, sancion in zip(lats, longs, sanciones)]

            mapObj_costo_calor = folium.Map(location=[meanLat, meanLong], zoom_start=14)
            HeatMap(heat_data_costo, radius=20, blur=15, max_zoom=1).add_to(mapObj_costo_calor)
            st_folium(mapObj_costo_calor, width=700, height=500)

        elif mapa_seleccionado == "Visualización de Clientes Reclamantes":
            st.write("### Visualización de Clientes Reclamantes")
            mapObj_clientes = folium.Map(location=[meanLat, meanLong], zoom_start=14)

            for _, row in df_filtrado.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lng']],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7,
                    tooltip=f"<b>Titular:</b> {row['TITULAR']}<br><b>Suministro:</b> {row['SUMINISTRO']}<br><b>Sanción Anual:</b> ${row['SANCION_ANUAL']:.2f}"
                ).add_to(mapObj_clientes)
            st_folium(mapObj_clientes, width=700, height=500)

        # Panel desplegable para estadísticas adicionales
        with st.expander("📊 Ver estadísticas adicionales"):
            st.write("### Resumen de Sanciones")
            sancion_total = df_filtrado['SANCION_ANUAL'].sum()
            sancion_promedio = df_filtrado['SANCION_ANUAL'].mean()
            st.metric(label="Sanción Total", value=f"${sancion_total:,.2f}")
            st.metric(label="Sanción Promedio por Cliente", value=f"${sancion_promedio:,.2f}")

            # Gráfico de barras de sanciones por distribuidor
            st.write("### Gráfico de Sanciones por Distribuidor")
            distribuidor_sancion = df_filtrado.groupby('NOM_ALIM')['SANCION_ANUAL'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            distribuidor_sancion.plot(kind='bar', ax=ax, color='teal')
            ax.set_title("Sanción Total por Distribuidor")
            ax.set_xlabel("Distribuidor")
            ax.set_ylabel("Sanción Total ($)")
            st.pyplot(fig)

    else:
        st.warning("Por favor, seleccione al menos un distribuidor para visualizar los mapas.")

def page_factibilidades():
    st.title("📄 Factibilidades de Suministro")
    subpages = ["Resumen", "Trazabilidad"]
    selected_page = st.radio("Seleccione una subpágina:", subpages)
    
    if selected_page == "Resumen":
        mostrar_resumen()
    elif selected_page == "Trazabilidad":
        mostrar_trazabilidad()

def mostrar_resumen():
    ruta_archivo_fact = 'Tablas/PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx'
    df_fact = cargar_datos_factibilidades(ruta_archivo_fact)
    
    # Convertir fechas a formato datetime
    df_fact['INGRESO'] = pd.to_datetime(df_fact['INGRESO'], errors='coerce')
    df_fact['EGRESO'] = pd.to_datetime(df_fact['EGRESO'], errors='coerce')
    
    # Filtros
    departamentos = df_fact['DEPARTAMENTO'].dropna().unique()
    selected_departamentos = st.multiselect("Seleccione uno o más Departamentos:", options=departamentos)
    
    solicitudes = df_fact['SOLICITUD'].dropna().unique()
    selected_solicitudes = st.multiselect("Seleccione uno o más Tipos de Solicitud:", options=solicitudes)
    
    tipo_solicitudes = df_fact['TIPO_SOLICITUD'].dropna().unique()
    selected_tipo_solicitudes = st.multiselect("Seleccione uno o más Tipos de Solicitud Específicos:", options=tipo_solicitudes)
    
    # Filtrado del DataFrame según las selecciones
    df_filtrado = df_fact.copy()
    if selected_departamentos:
        df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'].isin(selected_departamentos)]
    if selected_solicitudes:
        df_filtrado = df_filtrado[df_filtrado['SOLICITUD'].isin(selected_solicitudes)]
    if selected_tipo_solicitudes:
        df_filtrado = df_filtrado[df_filtrado['TIPO_SOLICITUD'].isin(selected_tipo_solicitudes)]
    
    st.write("### Datos Filtrados", df_filtrado)
    
    # Mapa interactivo si hay datos de latitud y longitud
    if 'latitud' in df_filtrado.columns and 'longitud' in df_filtrado.columns:
        st.write("### Mapa de Expedientes")
        df_mapa = df_filtrado.dropna(subset=['latitud', 'longitud'])
        fig = px.scatter_mapbox(df_mapa, lat='latitud', lon='longitud', hover_name='NOMBRE', 
                                hover_data=['DEPARTAMENTO', 'SOLICITUD', 'TIPO_SOLICITUD', 'INGRESO', 'EGRESO'], 
                                zoom=6, height=500)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No hay datos de ubicación disponibles para mostrar en el mapa.")

def mostrar_trazabilidad():
    st.write("### Trazabilidad del Expediente")
    ruta_archivo_fact = 'Tablas/PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx'
    df_fact = cargar_datos_factibilidades(ruta_archivo_fact)
    expediente_input = st.text_input("Ingrese el número de expediente:")
    
    if expediente_input:
        df_expediente = df_fact[df_fact['EXPEDIENTE'] == expediente_input]
        if not df_expediente.empty:
            trazabilidad = generar_trazabilidad(df_expediente)
            st.write(trazabilidad)
        else:
            st.write("No se encontraron registros para el expediente ingresado.")

def generar_trazabilidad(df_expediente):
    def format_date(dt):
        return f"{dt.day}/{dt.month}/{dt.year}"

    # Obtener el número de expediente
    expediente = df_expediente['EXPEDIENTE'].iloc[0]
    eventos = [f"**{expediente}**"]  # Título del expediente en negrita

    # Ordenar por fecha de ingreso para mantener el orden cronológico
    df_expediente = df_expediente.sort_values(by='INGRESO')

    # Iterar sobre cada fila para generar los eventos
    for _, row in df_expediente.iterrows():
        os_val = str(row['OS N°']).strip() if pd.notna(row['OS N°']) else ""
        os_val_upper = os_val.upper()

        if pd.notna(row['INGRESO']):
            eventos.append(f"- Ingreso a DPL: {format_date(row['INGRESO'])}")

        if os_val_upper.startswith("ENVIO OS"):
            numero_os = os_val.replace("ENVIO OS", "").strip()
            numero = f"N°{numero_os.replace('N', '').strip()}" if numero_os else ""
            if pd.notna(row['EGRESO']):
                eventos.append(f"- Envío de Orden de Servicio {numero} (DPL): {format_date(row['EGRESO'])}")
            if pd.notna(row['RESPUESTA_OS']):
                eventos.append(f"- Respuesta a Orden de Servicio {numero} (RT): {format_date(row['RESPUESTA_OS'])}")

        elif os_val_upper.startswith("PEDIDO COMERCIAL"):
            if pd.notna(row['EGRESO']):
                eventos.append(f"- Envío de Pedido Comercial (DPL): {format_date(row['EGRESO'])}")
            if pd.notna(row['RESPUESTA_OS']):
                eventos.append(f"- Respuesta a Pedido Comercial (RT): {format_date(row['RESPUESTA_OS'])}")

        elif os_val_upper.startswith("REINGRESO FINALIZADO"):
            if pd.notna(row['EGRESO']):
                eventos.append(f"- Finalización del Expediente (Reingreso): {format_date(row['EGRESO'])}")

        elif os_val_upper == "FINALIZADO":
            if pd.notna(row['EGRESO']):
                eventos.append(f"- Finalización del Expediente: {format_date(row['EGRESO'])}")

    # Convertir la lista de eventos en un string con saltos de línea
    trazabilidad = "\n".join(eventos)

    # Datos adicionales al final
    info_adicional = f"""
**Información adicional:**
- **Solicitud:** {df_expediente['SOLICITUD'].iloc[0]}, {df_expediente['TIPO_SOLICITUD'].iloc[0]}
- **Ubicación:** {df_expediente['latitud'].iloc[0]}, {df_expediente['longitud'].iloc[0]}
- **Obra de Infraestructura:** {df_expediente['Obra\nSolicitada'].iloc[0]} ({df_expediente['CODIGO OBRA '].iloc[0]})
- **Compuso:** {df_expediente['COMPUSO'].iloc[0]}
- **Representante Técnico:** {df_expediente['REPRESENTANTE\nTÉCNICO (RT)'].iloc[0]}
    """
    
    return trazabilidad + "\n\n" + info_adicional


# Main
if __name__ == "__main__":
    pages = {
        "👋 Bienvenida": page_bienvenida,
        "📊 Corriente por Distribuidor": page_corriente_dist,
        "🏭 Potencia por ET": page_potencia_et,
        "⚡ Corriente de LAT": page_corriente_lat,
        "🗺️ Mapa de Reclamos": page_mapa_reclamos,
        "📄 Factibilidades de Suministro": page_factibilidades
    }

    st.sidebar.title("Navegación")
    selected_page = st.sidebar.radio("Seleccione una página:", list(pages.keys()))
    pages[selected_page]()
