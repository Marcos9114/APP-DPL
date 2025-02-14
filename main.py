import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap, Draw
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
    """Carga datos y renombra columnas problemáticas."""
    df_fact = pd.read_excel(ruta_archivo)

    # Renombrar columnas para evitar caracteres especiales y espacios
    df_fact = df_fact.rename(columns={
        'Obra\nSolicitada': 'Obra_Solicitada',
        'CODIGO OBRA ': 'CODIGO_OBRA',
        'REPRESENTANTE\nTÉCNICO (RT)': 'REPRESENTANTE_TECNICO'
        # Añade aquí cualquier otro renombrado que necesites
    })
    return df_fact

def calcular_color(valor, max_valor):
    escala = int(255 * (valor / max_valor))
    return f"rgb({255 - escala}, {escala}, 100)"

def format_date(dt):
    return f"{dt.day}/{dt.month}/{dt.year}"

# Función para contar días hábiles entre dos fechas (excluye sábados y domingos)
def dias_habiles(start_date, end_date):
    days = 0
    current = start_date
    while current < end_date:
        current += datetime.timedelta(days=1)
        if current.weekday() < 5:  # 0: lunes, 6: domingo
            days += 1
    return days

# Función auxiliar para extraer el número de OS (ejemplo: "ENVIO OS N1" -> "N°1")
def extract_os_number(os_str):
    remainder = os_str[8:].strip()  # lo que sigue después de "ENVIO OS"
    if remainder:
        num = remainder.replace("N", "").strip()
        return f"N°{num}"
    return ""

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
    opciones_mapa = [
        "Mapa de Calor en función del Volumen de Reclamos", 
        "Mapa de Calor en función del Costo de Sanción", 
        "Visualización de Clientes Reclamantes"
    ]
    mapa_seleccionado = st.selectbox("Seleccione el mapa que desea visualizar:", opciones_mapa)

    # Multiselect para seleccionar distribuidores
    st.write("### Selección de Distribuidores:")
    distribuidor_unicos = sorted(df['NOM_ALIM'].unique())
    selected_distribuidores = st.multiselect("Seleccione uno o más distribuidores:", options=distribuidor_unicos)

    if selected_distribuidores:
        # Filtrar datos por distribuidores seleccionados y sanciones > 0
        df_filtrado = df[(df['NOM_ALIM'].isin(selected_distribuidores)) & (df['SANCION_ANUAL'] > 0)]
        lats = df_filtrado['lat'].tolist()
        longs = df_filtrado['lng'].tolist()
        sanciones = df_filtrado['SANCION_ANUAL'].tolist()

        # Calcular posición central para el mapa
        meanLat = statistics.mean(lats) if lats else -34.6037
        meanLong = statistics.mean(longs) if longs else -58.3816

        # Mostrar el mapa según la opción seleccionada
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

        # Estadísticas adicionales
        with st.expander("📊 Ver estadísticas adicionales"):
            st.write("### Resumen de Sanciones")
            sancion_total = df_filtrado['SANCION_ANUAL'].sum()
            sancion_promedio = df_filtrado['SANCION_ANUAL'].mean()
            st.metric(label="Sanción Total", value=f"${sancion_total:,.2f}")
            st.metric(label="Sanción Promedio por Cliente", value=f"${sancion_promedio:,.2f}")

            st.write("### Gráfico de Sanciones por Distribuidor")
            distribuidor_sancion = df_filtrado.groupby('NOM_ALIM')['SANCION_ANUAL'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            distribuidor_sancion.plot(kind='bar', ax=ax, color='teal')
            ax.set_title("Sanción Total por Distribuidor")
            ax.set_xlabel("Distribuidor")
            ax.set_ylabel("Sanción Total ($)")
            st.pyplot(fig)

        # Nueva funcionalidad: Seleccionar zona y calcular estadísticas de sanciones
        with st.expander("🖱️ Seleccionar zona y calcular estadísticas"):
            st.write("Seleccione un área en el mapa o ingrese las coordenadas manualmente para ver las estadísticas de sanciones en esa zona.")
            seleccion_manual = st.checkbox("Ingresar coordenadas manualmente", value=False)

            if not seleccion_manual:
                st.info("Utilice la herramienta de dibujo para seleccionar el área de interés.")
                # Crear un mapa con la herramienta de dibujo
                map_draw = folium.Map(location=[meanLat, meanLong], zoom_start=14)
                draw = Draw(export=True)
                draw.add_to(map_draw)
                # Mostrar el mapa con la herramienta de dibujo
                salida = st_folium(map_draw, width=700, height=500, key='draw_map')
                if salida and salida.get("all_drawings"):
                    dibujos = salida.get("all_drawings")
                    # Se usa el último dibujo realizado
                    ultimo_dibujo = dibujos[-1]
                    geometry = ultimo_dibujo.get("geometry", {})
                    if geometry.get("type") == "Polygon":
                        coords = geometry.get("coordinates", [])
                        # Se asume que se dibujó un rectángulo o polígono: se obtiene el bounding box
                        lats_sel = [p[1] for p in coords[0]]
                        lons_sel = [p[0] for p in coords[0]]
                        lat_min, lat_max = min(lats_sel), max(lats_sel)
                        lon_min, lon_max = min(lons_sel), max(lons_sel)
                        # Filtrar los datos dentro del área seleccionada
                        df_zona = df_filtrado[
                            (df_filtrado['lat'] >= lat_min) & (df_filtrado['lat'] <= lat_max) &
                            (df_filtrado['lng'] >= lon_min) & (df_filtrado['lng'] <= lon_max)
                        ]
                        sancion_total_zona = df_zona['SANCION_ANUAL'].sum()
                        sanciones_zona = df_zona.shape[0]
                        st.success(f"Sanción total en la zona seleccionada: ${sancion_total_zona:,.2f}")
                        st.success(f"Cantidad de sanciones en la zona: {sanciones_zona}")
                    else:
                        st.warning("El área dibujada no es un polígono válido.")
                else:
                    st.info("Dibuje un área en el mapa para ver las estadísticas.")
            else:
                st.write("Ingrese las coordenadas que delimitan el área de interés:")
                lat_min_input = st.number_input("Latitud mínima", value=meanLat-0.01, format="%.6f")
                lat_max_input = st.number_input("Latitud máxima", value=meanLat+0.01, format="%.6f")
                lon_min_input = st.number_input("Longitud mínima", value=meanLong-0.01, format="%.6f")
                lon_max_input = st.number_input("Longitud máxima", value=meanLong+0.01, format="%.6f")
                if st.button("Calcular estadísticas", key='calc_stats'):
                    df_zona_manual = df_filtrado[
                        (df_filtrado['lat'] >= lat_min_input) & (df_filtrado['lat'] <= lat_max_input) &
                        (df_filtrado['lng'] >= lon_min_input) & (df_filtrado['lng'] <= lon_max_input)
                    ]
                    sancion_total_zona_manual = df_zona_manual['SANCION_ANUAL'].sum()
                    sanciones_zona_manual = df_zona_manual.shape[0]
                    st.success(f"Sanción total en la zona seleccionada: ${sancion_total_zona_manual:,.2f}")
                    st.success(f"Cantidad de sanciones en la zona: {sanciones_zona_manual}")

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
    ruta_archivo_fact = r'Tablas/PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx'
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
    ruta_archivo_fact = r'Tablas/PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx'
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
    # Función auxiliar para formatear fechas sin ceros a la izquierda
    def format_date(dt):
        return f"{dt.day}/{dt.month}/{dt.year}"

    
    # Función auxiliar para extraer el número de OS (por ejemplo, "ENVIO OS N1" -> "N°1")
    def extract_os_number(os_str):
        remainder = os_str[8:].strip()  # lo que sigue después de "ENVIO OS"
        if remainder:
            num = remainder.replace("N", "").strip()
            return f"N°{num}"
        return ""
    
    # Obtener el número y el nombre del expediente (se asume que todas las filas son del mismo expediente)
    expediente = df_expediente['EXPEDIENTE'].iloc[0]
    nombre_expediente = df_expediente['NOMBRE'].iloc[0] if 'NOMBRE' in df_expediente.columns else ""
    result_lines = [f"Trazabilidad de Expediente: {expediente} - {nombre_expediente}"]
    
    
    # Ordenar cronológicamente según la columna INGRESO
    df_expediente = df_expediente.sort_values(by='INGRESO')
    
    # Variable que guarda la fecha de referencia para calcular los días hábiles
    ref_date = None

    for _, row in df_expediente.iterrows():
        os_val = str(row['OS N°']).strip() if pd.notna(row['OS N°']) else ""
        os_val_upper = os_val.upper()
        
        # Si aún no se estableció la fecha de referencia y el evento NO es un reingreso, se muestra "Ingreso a DPL"
        if ref_date is None and os_val_upper != "REINGRESO FINALIZADO":
            if pd.notna(row['INGRESO']):
                ref_date = row['INGRESO']
                result_lines.append(f"- Ingreso a DPL: {format_date(ref_date)}")
        
        # Caso: REINGRESO FINALIZADO
        if os_val_upper == "REINGRESO FINALIZADO":
            # Mostrar "Reingreso a DPL" usando la fecha de INGRESO
            if pd.notna(row['INGRESO']):
                reingreso_date = row['INGRESO']
                result_lines.append(f"- Reingreso a DPL: {format_date(reingreso_date)}")
            # Mostrar "Reingreso finalizado" usando la fecha de EGRESO y calcular días hábiles desde la fecha de reingreso
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(reingreso_date, row['EGRESO'])
                result_lines.append(f"- Reingreso finalizado: {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
        
        # Caso: ENVIO OS (Orden de Servicio)
        elif os_val_upper.startswith("ENVIO OS"):
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(ref_date, row['EGRESO']) if ref_date is not None else 0
                result_lines.append(f"- Envío de Orden de Servicio {extract_os_number(os_val)} (DPL): {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
            if pd.notna(row['RESPUESTA_OS']):
                dias = dias_habiles(ref_date, row['RESPUESTA_OS']) if ref_date is not None else 0
                result_lines.append(f"- Respuesta a Orden de Servicio {extract_os_number(os_val)} (RT): {format_date(row['RESPUESTA_OS'])} ({dias} días hábiles)")
                ref_date = row['RESPUESTA_OS']
        
        # Caso: PEDIDO COMERCIAL
        elif os_val_upper.startswith("PEDIDO COMERCIAL"):
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(ref_date, row['EGRESO']) if ref_date is not None else 0
                result_lines.append(f"- Pedido a Comercial (DPL): {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
            else:
                result_lines.append(f"- Pedido a Comercial (DPL): En proceso")
            if pd.notna(row['RESPUESTA_OS']):
                dias = dias_habiles(ref_date, row['RESPUESTA_OS']) if ref_date is not None else 0
                result_lines.append(f"- Respuesta de Comercial (RT): {format_date(row['RESPUESTA_OS'])} ({dias} días hábiles)")
                ref_date = row['RESPUESTA_OS']
            else:
                result_lines.append(f"- Respuesta de Comercial (RT): En proceso")
        
        # Caso: PEDIDO RELEVAM/DIGIT
        elif os_val_upper.startswith("PEDIDO RELEVAM") or os_val_upper.startswith("PEDIDO RELEVAM/DIGIT"):
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(ref_date, row['EGRESO']) if ref_date is not None else 0
                result_lines.append(f"- Pedido de Relevamiento/Digitalización (DPL): {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
            else:
                result_lines.append(f"- Pedido de Relevamiento/Digitalización (DPL): En proceso")
            if pd.notna(row['RESPUESTA_OS']):
                dias = dias_habiles(ref_date, row['RESPUESTA_OS']) if ref_date is not None else 0
                result_lines.append(f"- Relevamiento o digitalización efectuado (RT): {format_date(row['RESPUESTA_OS'])} ({dias} días hábiles)")
                ref_date = row['RESPUESTA_OS']
            else:
                result_lines.append(f"- Relevamiento o digitalización efectuado (RT): En proceso")
        
        # Caso: REVISION DNC
        elif os_val_upper.startswith("REVISION DNC"):
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(ref_date, row['EGRESO']) if ref_date is not None else 0
                result_lines.append(f"- Revisión DNC (DPL): {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
            else:
                result_lines.append(f"- Revisión DNC (DPL): En proceso")
            if pd.notna(row['RESPUESTA_OS']):
                dias = dias_habiles(ref_date, row['RESPUESTA_OS']) if ref_date is not None else 0
                result_lines.append(f"- Respuesta a Revisión DNC (RT): {format_date(row['RESPUESTA_OS'])} ({dias} días hábiles)")
                ref_date = row['RESPUESTA_OS']
            else:
                result_lines.append(f"- Respuesta a Revisión DNC (RT): En proceso")
        
        # Caso: FINALIZADO
        elif os_val_upper == "FINALIZADO":
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(ref_date, row['EGRESO']) if ref_date is not None else 0
                result_lines.append(f"- Finalización del Expediente: {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
            else:
                result_lines.append(f"- Finalización del Expediente: En proceso")
    
    trazabilidad_text = "\n".join(result_lines)
    
    # Información adicional: se toma la fila donde OS N° sea "FINALIZADO" (última ocurrencia)
    df_finalizado = df_expediente[df_expediente['OS N°'].astype(str).str.upper() == "FINALIZADO"]
    if not df_finalizado.empty:
        fila_finalizado = df_finalizado.iloc[-1]
    else:
        fila_finalizado = None

    def get_info(col):
        if fila_finalizado is not None and pd.notna(fila_finalizado.get(col, None)):
            return fila_finalizado[col]
        else:
            return "----"

    info_adicional = f"""

**Información adicional:**
- **Solicitud:** {get_info('SOLICITUD')}, {get_info('TIPO_SOLICITUD')}
- **Ubicación:** {get_info('latitud')}, {get_info('longitud')}
- **Obra de Infraestructura:** {get_info('Obra Solicitada')} ({get_info('CODIGO OBRA ')})
- **Compuso:** {get_info('COMPUSO')}
- **Representante Técnico:** {get_info('REPRESENTANTE TÉCNICO (RT)')}
- **Potencia [kW]:** {get_info('POTENCIA')}
- **Distribuidor:** {get_info('DISTRIBUIDOR')}
- **ET_CD:** {get_info('ET_CD')}
- **Comentarios adicionales:** {get_info('MOTIVO DEMORA')}
    """
    
    return trazabilidad_text + "\n\n" + info_adicional

# Función para mostrar la trazabilidad con búsqueda por expediente (no case sensitive)
def mostrar_trazabilidad():
    st.write("### Trazabilidades")
    ruta_archivo_fact = r"Tablas/PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx"
    df_fact = cargar_datos_factibilidades(ruta_archivo_fact)
    
    search_input = st.text_input("Ingrese parte del número o nombre del expediente:")
    if search_input:
        # Buscar coincidencias tanto en la columna 'EXPEDIENTE' como en 'NOMBRE'
        mask = df_fact['EXPEDIENTE'].astype(str).str.contains(search_input, case=False, na=False) | \
               df_fact['NOMBRE'].astype(str).str.contains(search_input, case=False, na=False)
        df_coincidencias = df_fact[mask]
        if df_coincidencias.empty:
            st.write("No se encontraron coincidencias.")
        else:
            opciones = [f"{row['EXPEDIENTE']} - {row['NOMBRE']}" for _, row in df_coincidencias.iterrows()]
            selected_option = st.selectbox("Seleccione un expediente:", opciones)
            expediente_num = selected_option.split(" - ")[0]
            df_expediente = df_fact[df_fact['EXPEDIENTE'].astype(str) == expediente_num]
            #st.markdown(f"<h3 style='font-size:20px;'>{selected_option}</h3>", unsafe_allow_html=True)
            trazabilidad = generar_trazabilidad(df_expediente)
            st.text(trazabilidad)
    else:
        st.write("Ingrese parte del número o nombre del expediente para buscar.")

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
