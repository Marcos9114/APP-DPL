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
import numpy as np
from scipy.stats import norm
import scipy.stats as stats

# Importar ARIMA para el pronóstico (modelo de series de tiempo)
from statsmodels.tsa.arima.model import ARIMA

# Configuración inicial de la página
st.set_page_config(
    page_title="Sistema de Monitoreo de Corriente",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
    /* Color de fondo de la barra lateral */
    .css-1d391kg {background-color: #f0f2f6;}
    /* Colores para títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #4a90e2;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

@st.cache_data
def cargar_datos(ruta):
    return pd.read_parquet(ruta)

@st.cache_data
def cargar_datos_factibilidades(ruta_archivo):
    df_fact = pd.read_excel(ruta_archivo)
    df_fact = df_fact.rename(columns={
        'Obra\nSolicitada': 'Obra_Solicitada',
        'CODIGO OBRA ': 'CODIGO_OBRA',
        'REPRESENTANTE\nTÉCNICO (RT)': 'REPRESENTANTE_TECNICO'
    })
    return df_fact

def calcular_color(valor, max_valor):
    escala = int(255 * (valor / max_valor))
    return f"rgb({255 - escala}, {escala}, 100)"

def format_date(dt):
    return f"{dt.day}/{dt.month}/{dt.year}"

def dias_habiles(start_date, end_date):
    days = 0
    current = start_date
    while current < end_date:
        current += datetime.timedelta(days=1)
        if current.weekday() < 5:
            days += 1
    return days

def extract_os_number(os_str):
    remainder = os_str[8:].strip()
    if remainder:
        num = remainder.replace("N", "").strip()
        return f"N°{num}"
    return ""

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

# ----------------------- Páginas de la aplicación -----------------------

def page_bienvenida():
    st.title("👋 Bienvenido al Sistema de Monitoreo de Corriente")
    st.write("""
        Esta aplicación permite visualizar y analizar las mediciones de corriente y temperatura, 
        realizando estudios probabilísticos y pronósticos que apoyan en la toma de decisiones:
        - **Corriente por Distribuidor:** Análisis estadístico y probabilístico de la corriente.
        - **Potencia por ET:** Mediciones de potencia por Estación Transformadora.
        - **Corriente de LAT:** Mediciones en líneas de Alta Tensión.
        - **Mapa de Reclamos:** Visualización de sanciones y reclamos.
        - **Factibilidades:** Resumen y búsqueda de expedientes.
        - **Pronóstico de Demanda:** Pronóstico de la corriente usando modelos ARIMA.
    """)

def page_corriente_dist():
    st.title("📊 Corriente por Distribuidor")
    ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
    df = cargar_datos(ruta_archivo)
    alim_unicos = df['ALIM'].unique()
    selected_alim = st.multiselect("Seleccione uno o más alimentadores (ALIM):", options=alim_unicos)
    
    if selected_alim:
        filtered_df = df[df['ALIM'].isin(selected_alim)].sort_values(by='TIME')
        # Asegurarse que la columna TIME sea datetime
        filtered_df['TIME'] = pd.to_datetime(filtered_df['TIME'])
        
        st.subheader("Serie Temporal de Corriente")
        fig_corriente = crear_grafico_linea(
            filtered_df, 'TIME', 'VALUE', 'ALIM', 
            "Evolución Horaria de la Corriente", 
            "Fecha y Hora", "Valor de Corriente", "Alimentador"
        )
        st.plotly_chart(fig_corriente, use_container_width=False)
        st.write("Esta gráfica muestra la evolución de la corriente a lo largo del tiempo para el/los alimentador(es) seleccionado(s).")
        
        st.markdown("---")
        
        # Histograma con curva de densidad
        st.subheader("Histograma y Curva de Densidad")
        current_values = filtered_df['VALUE']
        mu, std = current_values.mean(), current_values.std()
        hist_fig = px.histogram(filtered_df, x='VALUE', histnorm='density', nbins=50,
                                title="Distribución de Corriente")
        x_vals = np.linspace(current_values.min(), current_values.max(), 100)
        y_vals = norm.pdf(x_vals, mu, std)
        hist_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Curva Gaussiana'))
        st.plotly_chart(hist_fig, use_container_width=False)
        st.write("El histograma con la curva de densidad te permite visualizar si la distribución de la corriente se asemeja a una distribución normal. Un buen ajuste indica que la mayoría de los valores se concentran alrededor de la media, facilitando la detección de anomalías.")
        
        st.markdown("---")
        
        # Q-Q Plot
        st.subheader("Gráfico Q-Q")
        fig_qq, ax = plt.subplots(figsize=(6,6))
        stats.probplot(current_values, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot de la Corriente")
        st.pyplot(fig_qq)
        st.write("El gráfico Q-Q compara la distribución empírica de la corriente con una distribución normal. Si los puntos se alinean sobre la línea teórica, se confirma un buen ajuste a la normalidad.")
        
        st.markdown("---")
        
        # Serie Temporal con Bandas de Confianza
        st.subheader("Serie Temporal con Bandas de Confianza")
        df_ts = filtered_df.set_index('TIME').sort_index()
        window = 24  # ventana de 24 horas (media móvil diaria)
        df_ts['rolling_mean'] = df_ts['VALUE'].rolling(window=window).mean()
        df_ts['rolling_std'] = df_ts['VALUE'].rolling(window=window).std()
        df_ts['upper_band'] = df_ts['rolling_mean'] + 2 * df_ts['rolling_std']
        df_ts['lower_band'] = df_ts['rolling_mean'] - 2 * df_ts['rolling_std']
        
        fig_bands = go.Figure()
        fig_bands.add_trace(go.Scatter(x=df_ts.index, y=df_ts['VALUE'], mode='lines', name='Corriente'))
        fig_bands.add_trace(go.Scatter(x=df_ts.index, y=df_ts['rolling_mean'], mode='lines', name='Media Móvil', line=dict(color='orange')))
        fig_bands.add_trace(go.Scatter(x=df_ts.index, y=df_ts['upper_band'], mode='lines', name='Banda Superior', line=dict(dash='dash', color='green')))
        fig_bands.add_trace(go.Scatter(x=df_ts.index, y=df_ts['lower_band'], mode='lines', name='Banda Inferior', line=dict(dash='dash', color='red')))
        fig_bands.update_layout(
            title="Serie Temporal con Bandas de Confianza (±2σ)",
            xaxis_title="Fecha y Hora",
            yaxis_title="Valor de Corriente",
            template='plotly_dark',
            width=2000,
            height=600
        )
        st.plotly_chart(fig_bands, use_container_width=False)
        st.write("Esta visualización muestra la serie histórica junto con la media móvil y bandas de ±2 desviaciones estándar. Las bandas ayudan a identificar anomalías o comportamientos atípicos en la corriente.")
        
        st.markdown("---")
        
        # Scatter plot de Corriente vs Temperatura
        st.subheader("Correlación entre Corriente y Temperatura")
        ruta_temperatura = 'Tablas/mendoza_aero.parquet'
        df_temp = cargar_datos(ruta_temperatura)
        df_temp['FECHA'] = pd.to_datetime(df_temp['FECHA'])
        # Convertir la columna TEMP a numérico, ignorando errores
        df_temp['TEMP'] = pd.to_numeric(df_temp['TEMP'], errors='coerce')
        df_temp = df_temp.sort_values('FECHA')

        df_ts_reset = df_ts.reset_index()
        # Hacer merge asof con tolerancia de 1 hora
        df_merged = pd.merge_asof(
        df_ts_reset.sort_values('TIME'),
        df_temp.sort_values('FECHA'),
        left_on='TIME',
        right_on='FECHA',
        tolerance=pd.Timedelta('1h')
        )
        df_merged = df_merged.dropna(subset=['TEMP'])

        fig_scatter = px.scatter(df_merged, x='TEMP', y='VALUE', trendline='ols',
                         title="Relación entre Temperatura y Corriente")
        st.plotly_chart(fig_scatter, use_container_width=True)
        corr_coeff = df_merged['TEMP'].corr(df_merged['VALUE'])
        st.write(f"El coeficiente de correlación entre la temperatura y la corriente es: {corr_coeff:.2f}.")
        
    else:
        st.write("Por favor, seleccione al menos un alimentador (ALIM).")

# Página: Pronóstico de Demanda usando ARIMA y análisis de volatilidad
def page_pronostico_demanda():
    st.title("🔮 Pronóstico de Demanda de Corriente")
    st.write("Esta página utiliza un modelo ARIMA para pronosticar la demanda de corriente a futuro, permitiendo anticipar picos o caídas y analizar la volatilidad en la serie. Esto es clave para la planificación y toma de decisiones en estudios eléctricos.")
    
    ruta_archivo = 'Tablas/corriente_alimentador_2024.parquet'
    df = cargar_datos(ruta_archivo)
    alim_unicos = df['ALIM'].unique()
    selected_alim = st.selectbox("Seleccione un alimentador (ALIM) para el pronóstico:", options=alim_unicos)
    
    if selected_alim:
        df_alim = df[df['ALIM'] == selected_alim].copy()
        df_alim['TIME'] = pd.to_datetime(df_alim['TIME'])
        df_alim = df_alim.sort_values(by='TIME')
        df_alim.set_index('TIME', inplace=True)
        
        st.write("Ajustando el modelo ARIMA(1,1,1)... Esto puede tardar unos segundos.")
        try:
            model = ARIMA(df_alim['VALUE'], order=(1,1,1))
            model_fit = model.fit()
            forecast_steps = st.number_input("Número de horas a pronosticar:", min_value=1, max_value=168, value=24)
            forecast_result = model_fit.get_forecast(steps=forecast_steps)
            forecast_df = forecast_result.conf_int()
            forecast_df['forecast'] = forecast_result.predicted_mean
            forecast_df.index = pd.date_range(start=df_alim.index[-1] + pd.Timedelta(hours=1), periods=forecast_steps, freq='H')
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=df_alim.index, y=df_alim['VALUE'], mode='lines', name='Histórico'))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['forecast'], mode='lines', name='Pronóstico'))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['upper VALUE'], mode='lines', name='Límite Superior', line=dict(dash='dash')))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['lower VALUE'], mode='lines', name='Límite Inferior', line=dict(dash='dash')))
            fig_forecast.update_layout(
                title=f"Pronóstico de Demanda de Corriente para {selected_alim}",
                xaxis_title="Fecha y Hora",
                yaxis_title="Valor de Corriente",
                template='plotly_dark',
                width=2000,
                height=600
            )
            st.plotly_chart(fig_forecast, use_container_width=False)
            st.write(f"El gráfico muestra la serie histórica y el pronóstico para las próximas {forecast_steps} horas. Los intervalos de confianza (límites superior e inferior) indican la volatilidad esperada y permiten planificar acciones en función de posibles escenarios.")
        except Exception as e:
            st.error(f"Error al ajustar el modelo ARIMA: {e}")
    else:
        st.write("Por favor, seleccione un alimentador (ALIM) para continuar.")

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
            fig_et = crear_grafico_linea(
                filtered_df_et, 'TIME', 'VALUE', 'TRAFO_LABEL', 
                "Mediciones de Potencia [kW] por ET y Transformador", 
                "Fecha y Hora", "Valor de Potencia [kW]", "Transformador"
            )
            st.plotly_chart(fig_et, use_container_width=False)
        else:
            st.write("Por favor, seleccione al menos un transformador (TRAFO).")
    else:
        st.write("Por favor, seleccione al menos una estación transformadora (ET).")

def page_corriente_lat():
    st.title("⚡ Corriente de Líneas de Alta Tensión (LAT)")
    ruta_archivo_lat = 'Tablas/corriente_LAT_2024.parquet'
    df_lat = cargar_datos(ruta_archivo_lat)
    lat_unicos = df_lat['NOMBRE'].unique()
    selected_lat = st.multiselect("Seleccione una o más Líneas de Alta Tensión (LAT):", options=lat_unicos)
    
    if selected_lat:
        filtered_df_lat = df_lat[df_lat['NOMBRE'].isin(selected_lat)].sort_values(by='TIME')
        fig_lat = crear_grafico_linea(
            filtered_df_lat, 'TIME', 'VALUE', 'NOMBRE', 
            "Mediciones de Corriente por Línea de Alta Tensión (LAT)", 
            "Fecha y Hora", "Valor de Corriente", "Línea de Alta Tensión"
        )
        st.plotly_chart(fig_lat, use_container_width=False)
    else:
        st.write("Por favor, seleccione al menos una línea de alta tensión (LAT).")

def page_mapa_reclamos():
    st.title("🗺️ Mapas de Sanciones por Distribuidor")
    ruta_archivo = 'Tablas/SANCIONES_30_31_x_y.parquet'
    df = cargar_datos(ruta_archivo)
    
    st.write("### Selección de visualización:")
    opciones_mapa = [
        "Mapa de Calor en función del Volumen de Reclamos", 
        "Mapa de Calor en función del Costo de Sanción", 
        "Visualización de Clientes Reclamantes"
    ]
    mapa_seleccionado = st.selectbox("Seleccione el mapa que desea visualizar:", opciones_mapa)
    
    st.write("### Selección de Distribuidores:")
    distribuidor_unicos = sorted(df['NOM_ALIM'].unique())
    selected_distribuidores = st.multiselect("Seleccione uno o más distribuidores:", options=distribuidor_unicos)
    
    if selected_distribuidores:
        df_filtrado = df[(df['NOM_ALIM'].isin(selected_distribuidores)) & (df['SANCION_ANUAL'] > 0)]
        lats = df_filtrado['lat'].tolist()
        longs = df_filtrado['lng'].tolist()
        sanciones = df_filtrado['SANCION_ANUAL'].tolist()
        
        meanLat = statistics.mean(lats) if lats else -34.6037
        meanLong = statistics.mean(longs) if longs else -58.3816
        
        if mapa_seleccionado == "Mapa de Calor en función del Volumen de Reclamos":
            st.write("### Mapa de Calor (Volumen de Reclamos)")
            ubicaciones = defaultdict(int)
            for lat, lng in zip(lats, longs):
                ubicaciones[(lat, lng)] += 1
            heat_data = [[lat, lng, count] for (lat, lng), count in ubicaciones.items()]
            mapObj_calor = folium.Map(location=[meanLat, meanLong], zoom_start=14)
            HeatMap(heat_data, radius=15, blur=20, max_zoom=1).add_to(mapObj_calor)
            st_folium(mapObj_calor, width=700, height=500)
        
        elif mapa_seleccionado == "Mapa de Calor en función del Costo de Sanción":
            st.write("### Mapa de Calor (Costo de Sanción)")
            heat_data_costo = [[lat, lng, sancion] for lat, lng, sancion in zip(lats, longs, sanciones)]
            mapObj_costo_calor = folium.Map(location=[meanLat, meanLong], zoom_start=14)
            HeatMap(heat_data_costo, radius=20, blur=15, max_zoom=1).add_to(mapObj_costo_calor)
            st_folium(mapObj_costo_calor, width=700, height=500)
        
        elif mapa_seleccionado == "Visualización de Clientes Reclamantes":
            st.write("### Clientes Reclamantes")
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
        
        with st.expander("📊 Ver estadísticas adicionales"):
            st.write("### Resumen de Sanciones")
            sancion_total = df_filtrado['SANCION_ANUAL'].sum()
            sancion_promedio = df_filtrado['SANCION_ANUAL'].mean()
            st.metric(label="Sanción Total", value=f"${sancion_total:,.2f}")
            st.metric(label="Sanción Promedio", value=f"${sancion_promedio:,.2f}")
            
            st.write("### Gráfico de Sanciones por Distribuidor")
            distribuidor_sancion = df_filtrado.groupby('NOM_ALIM')['SANCION_ANUAL'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            distribuidor_sancion.plot(kind='bar', ax=ax, color='teal')
            ax.set_title("Sanción Total por Distribuidor")
            ax.set_xlabel("Distribuidor")
            ax.set_ylabel("Sanción Total ($)")
            st.pyplot(fig)
        
        with st.expander("🖱️ Seleccionar zona y calcular estadísticas"):
            st.write("Seleccione un área en el mapa o ingrese las coordenadas manualmente para ver estadísticas de sanciones.")
            seleccion_manual = st.checkbox("Ingresar coordenadas manualmente", value=False)
            if not seleccion_manual:
                st.info("Utilice la herramienta de dibujo para seleccionar el área de interés.")
                map_draw = folium.Map(location=[meanLat, meanLong], zoom_start=14)
                draw = Draw(export=True)
                draw.add_to(map_draw)
                salida = st_folium(map_draw, width=700, height=500, key='draw_map')
                if salida and salida.get("all_drawings"):
                    dibujos = salida.get("all_drawings")
                    ultimo_dibujo = dibujos[-1]
                    geometry = ultimo_dibujo.get("geometry", {})
                    if geometry.get("type") == "Polygon":
                        coords = geometry.get("coordinates", [])
                        lats_sel = [p[1] for p in coords[0]]
                        lons_sel = [p[0] for p in coords[0]]
                        lat_min, lat_max = min(lats_sel), max(lats_sel)
                        lon_min, lon_max = min(lons_sel), max(lons_sel)
                        df_zona = df_filtrado[
                            (df_filtrado['lat'] >= lat_min) & (df_filtrado['lat'] <= lat_max) &
                            (df_filtrado['lng'] >= lon_min) & (df_filtrado['lng'] <= lon_max)
                        ]
                        sancion_total_zona = df_zona['SANCION_ANUAL'].sum()
                        sanciones_zona = df_zona.shape[0]
                        st.success(f"Sanción total en la zona: ${sancion_total_zona:,.2f}")
                        st.success(f"Cantidad de sanciones: {sanciones_zona}")
                    else:
                        st.warning("El área dibujada no es un polígono válido.")
                else:
                    st.info("Dibuje un área en el mapa para ver estadísticas.")
            else:
                st.write("Ingrese las coordenadas del área:")
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
                    st.success(f"Sanción total en la zona: ${sancion_total_zona_manual:,.2f}")
                    st.success(f"Cantidad de sanciones: {sanciones_zona_manual}")
    else:
        st.warning("Por favor, seleccione al menos un distribuidor para visualizar los mapas.")

# Función para generar la trazabilidad de un expediente (usada en Buscador)
def generar_trazabilidad(df_expediente):
    expediente = df_expediente['EXPEDIENTE'].iloc[0]
    nombre_expediente = df_expediente['NOMBRE'].iloc[0] if 'NOMBRE' in df_expediente.columns else ""
    result_lines = [f"Trazabilidad de Expediente: {expediente} - {nombre_expediente}"]
    
    df_expediente = df_expediente.sort_values(by='INGRESO')
    ref_date = None

    for _, row in df_expediente.iterrows():
        os_val = str(row['OS N°']).strip() if pd.notna(row['OS N°']) else ""
        os_val_upper = os_val.upper()
        if ref_date is None and os_val_upper != "REINGRESO FINALIZADO":
            if pd.notna(row['INGRESO']):
                ref_date = row['INGRESO']
                result_lines.append(f"- Ingreso a DPL: {format_date(ref_date)}")
        if os_val_upper == "REINGRESO FINALIZADO":
            if pd.notna(row['INGRESO']):
                reingreso_date = row['INGRESO']
                result_lines.append(f"- Reingreso a DPL: {format_date(reingreso_date)}")
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(reingreso_date, row['EGRESO'])
                result_lines.append(f"- Reingreso finalizado: {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
        elif os_val_upper.startswith("ENVIO OS"):
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(ref_date, row['EGRESO']) if ref_date is not None else 0
                result_lines.append(f"- Envío de Orden de Servicio {extract_os_number(os_val)} (DPL): {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
            if pd.notna(row['RESPUESTA_OS']):
                dias = dias_habiles(ref_date, row['RESPUESTA_OS']) if ref_date is not None else 0
                result_lines.append(f"- Respuesta a Orden de Servicio {extract_os_number(os_val)} (RT): {format_date(row['RESPUESTA_OS'])} ({dias} días hábiles)")
                ref_date = row['RESPUESTA_OS']
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
        elif os_val_upper == "FINALIZADO":
            if pd.notna(row['EGRESO']):
                dias = dias_habiles(ref_date, row['EGRESO']) if ref_date is not None else 0
                result_lines.append(f"- Finalización del Expediente: {format_date(row['EGRESO'])} ({dias} días hábiles)")
                ref_date = row['EGRESO']
            else:
                result_lines.append(f"- Finalización del Expediente: En proceso")
    
    trazabilidad_text = "\n".join(result_lines)
    df_finalizado = df_expediente[df_expediente['OS N°'].astype(str).str.upper() == "FINALIZADO"]
    fila_finalizado = df_finalizado.iloc[-1] if not df_finalizado.empty else None

    def get_info(col):
        if fila_finalizado is not None and pd.notna(fila_finalizado.get(col, None)):
            return fila_finalizado[col]
        else:
            return "----"

    info_adicional = f"""

**Información adicional:**
- **Solicitud:** {get_info('SOLICITUD')}, {get_info('TIPO_SOLICITUD')}
- **Ubicación:** {get_info('latitud')}, {get_info('longitud')}
- **Obra de Infraestructura:** {get_info('Obra_Solicitada')} ({get_info('CODIGO_OBRA')})
- **Compuso:** {get_info('COMPUSO')}
- **Representante Técnico:** {get_info('REPRESENTANTE_TECNICO')}
- **Potencia [kW]:** {get_info('POTENCIA')}
- **Distribuidor:** {get_info('DISTRIBUIDOR')}
- **ET_CD:** {get_info('ET_CD')}
- **Comentarios adicionales:** {get_info('MOTIVO DEMORA')}
    """
    return trazabilidad_text + "\n\n" + info_adicional

def page_buscador_expedientes():
    st.title("🔍 Buscador de Expedientes")
    ruta_archivo_fact = r"Tablas/PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx"
    df_fact = cargar_datos_factibilidades(ruta_archivo_fact)
    
    search_input = st.text_input("Ingrese parte del número o nombre del expediente para buscar:")
    if search_input:
        mask = df_fact['EXPEDIENTE'].astype(str).str.contains(search_input, case=False, na=False) | \
               df_fact['NOMBRE'].astype(str).str.contains(search_input, case=False, na=False)
        df_coincidencias = df_fact[mask]
        if df_coincidencias.empty:
            st.info("No se encontraron coincidencias.")
        else:
            opciones = [f"{row['EXPEDIENTE']} - {row['NOMBRE']}" for _, row in df_coincidencias.iterrows()]
            selected_option = st.selectbox("Seleccione un expediente:", opciones)
            expediente_num = selected_option.split(" - ")[0]
            df_expediente = df_fact[df_fact['EXPEDIENTE'].astype(str) == expediente_num]
            trazabilidad = generar_trazabilidad(df_expediente)
            st.markdown(f"```\n{trazabilidad}\n```")
    else:
        st.write("Ingrese parte del número o nombre del expediente para buscar.")

def page_resumen_factibilidades():
    st.title("📄 Resumen de Factibilidades")
    ruta_archivo_fact = r'Tablas/PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx'
    df_fact = cargar_datos_factibilidades(ruta_archivo_fact)
    
    # Conversión de fechas
    df_fact['INGRESO'] = pd.to_datetime(df_fact['INGRESO'], errors='coerce')
    df_fact['EGRESO'] = pd.to_datetime(df_fact['EGRESO'], errors='coerce')
    
    # Filtros interactivos
    departamentos = df_fact['DEPARTAMENTO'].dropna().unique()
    selected_departamentos = st.multiselect("Seleccione Departamentos:", options=departamentos)
    solicitudes = df_fact['SOLICITUD'].dropna().unique()
    selected_solicitudes = st.multiselect("Seleccione Tipos de Solicitud:", options=solicitudes)
    tipo_solicitudes = df_fact['TIPO_SOLICITUD'].dropna().unique()
    selected_tipo_solicitudes = st.multiselect("Seleccione Tipos de Solicitud Específicos:", options=tipo_solicitudes)
    
    df_filtrado = df_fact.copy()
    if selected_departamentos:
        df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'].isin(selected_departamentos)]
    if selected_solicitudes:
        df_filtrado = df_filtrado[df_filtrado['SOLICITUD'].isin(selected_solicitudes)]
    if selected_tipo_solicitudes:
        df_filtrado = df_filtrado[df_filtrado['TIPO_SOLICITUD'].isin(selected_tipo_solicitudes)]
    
    st.write("### Datos Filtrados", df_filtrado)
    
    if 'latitud' in df_filtrado.columns and 'longitud' in df_filtrado.columns:
        st.write("### Mapa de Expedientes")
        df_mapa = df_filtrado.dropna(subset=['latitud', 'longitud'])
        fig = px.scatter_mapbox(
            df_mapa, lat='latitud', lon='longitud', hover_name='NOMBRE', 
            hover_data=['DEPARTAMENTO', 'SOLICITUD', 'TIPO_SOLICITUD', 'INGRESO', 'EGRESO'], 
            zoom=6, height=500
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No hay datos de ubicación para mostrar en el mapa.")

def page_analisis_factibilidades():
    st.title("📈 Análisis de Factibilidades")
    ruta_archivo_fact = r'Tablas/PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx'
    df_fact = cargar_datos_factibilidades(ruta_archivo_fact)
    
    st.write("### Vista preliminar de datos")
    st.dataframe(df_fact.head())
    
    if 'DEPARTAMENTO' in df_fact.columns:
        depto_count = df_fact['DEPARTAMENTO'].value_counts().reset_index()
        depto_count.columns = ['DEPARTAMENTO', 'Cantidad']
        fig = px.bar(depto_count, x='DEPARTAMENTO', y='Cantidad', 
                     title="Cantidad de Expedientes por Departamento", 
                     color='Cantidad', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    if 'SOLICITUD' in df_fact.columns:
        solicitud_count = df_fact['SOLICITUD'].value_counts().reset_index()
        solicitud_count.columns = ['SOLICITUD', 'Cantidad']
        fig2 = px.pie(solicitud_count, names='SOLICITUD', values='Cantidad', 
                      title="Distribución de Tipos de Solicitud")
        st.plotly_chart(fig2, use_container_width=True)

def page_analisis_sanciones():
    st.title("📉 Análisis de Sanciones")
    ruta_archivo = 'Tablas/SANCIONES_30_31_x_y.parquet'
    df = cargar_datos(ruta_archivo)
    
    st.write("### Datos de Sanciones")
    st.dataframe(df.head())
    
    if 'NOM_ALIM' in df.columns:
        sancion_sum = df.groupby('NOM_ALIM')['SANCION_ANUAL'].sum().reset_index()
        fig = px.bar(sancion_sum, x='NOM_ALIM', y='SANCION_ANUAL',
                     title="Sanción Total por Distribuidor", 
                     color='SANCION_ANUAL', color_continuous_scale='Bluered')
        st.plotly_chart(fig, use_container_width=True)

# ----------------------- Menú de Navegación -----------------------

pages = {
    "👋 Bienvenida": page_bienvenida,
    "📊 Corriente por Distribuidor": page_corriente_dist,
    "🔮 Pronóstico de Demanda": page_pronostico_demanda,
    "🏭 Potencia por ET": page_potencia_et,
    "⚡ Corriente de LAT": page_corriente_lat,
    "🗺️ Mapa de Reclamos": page_mapa_reclamos,
    "📄 Resumen de Factibilidades": page_resumen_factibilidades,
    "🔍 Buscador de Expedientes": page_buscador_expedientes,
    "📈 Análisis de Factibilidades": page_analisis_factibilidades,
    "📉 Análisis de Sanciones": page_analisis_sanciones

}

st.sidebar.title("Navegación")
selected_page = st.sidebar.radio("Seleccione una página:", list(pages.keys()))
pages[selected_page]()
