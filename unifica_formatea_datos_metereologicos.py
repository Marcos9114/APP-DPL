import os
import pandas as pd

ruta_carpeta = r"C:\Users\marco\OneDrive\Escritorio\APPWEB-DPL\APP-DPL\Datos_meteorológicos"
archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith(".txt")]

# Lista para almacenar todos los DataFrames
dfs = []

for archivo in archivos:
    ruta_completa = os.path.join(ruta_carpeta, archivo)
    
    # Leer el archivo con codificación Johab
    try:
        with open(ruta_completa, 'r', encoding='johab') as f:
            lineas = f.readlines()
    except UnicodeDecodeError:
        print(f"Error de codificación en {archivo}. Probando alternativas...")
        with open(ruta_completa, 'r', encoding='latin-1') as f:
            lineas = f.readlines()

    # Procesar las líneas
    datos = []
    for linea in lineas[2:]:  # Saltar encabezados
        if len(linea.strip()) == 0:
            continue
        
        # Extraer campos
        fecha = linea[0:8].strip()
        hora = linea[8:14].strip()
        temp = linea[14:20].strip()
        hum = linea[20:26].strip()
        pnm = linea[26:34].strip()
        dd = linea[34:40].strip()
        ff = linea[40:48].strip()
        nombre = linea[48:].strip()

        datos.append([fecha, hora, temp, hum, pnm, dd, ff, nombre])

    # Crear DataFrame
    columnas = ['FECHA', 'HORA', 'TEMP', 'HUM', 'PNM', 'DD', 'FF', 'NOMBRE']
    df = pd.DataFrame(datos, columns=columnas)
    dfs.append(df)

# Unificar todos los DataFrames
df_final = pd.concat(dfs, ignore_index=True)

# Filtrar estaciones específicas
estaciones = ["MENDOZA AERO", "MENDOZA OBSERVATORIO", "SAN RAFAEL AERO", "USPALLATA"]
df_filtrado = df_final[df_final['NOMBRE'].isin(estaciones)]

# Convertir FECHA (DDMMYYYY) y HORA a datetime
df_filtrado['FECHA'] = (
    pd.to_datetime(df_filtrado['FECHA'], format='%d%m%Y', errors='coerce') +  # Convertir a fecha
    pd.to_timedelta(pd.to_numeric(df_filtrado['HORA'], errors='coerce').fillna(0).astype(int), unit='h')  # Agregar horas
)

# Eliminar la columna HORA y reordenar
df_filtrado = df_filtrado.drop(columns=['HORA'])
df_filtrado = df_filtrado[['FECHA', 'TEMP', 'HUM', 'PNM', 'DD', 'FF', 'NOMBRE']]

# Ordenar por FECHA (de más antiguo a más reciente)
df_filtrado = df_filtrado.sort_values(by='FECHA')

# Guardar archivo con todos los datos filtrados
df_filtrado.to_parquet(
    'datos_filtrados.parquet',
    engine='pyarrow',
    compression='snappy'
)

# Filtrar solo "MENDOZA AERO"
df_mendoza_aero = df_filtrado[df_filtrado['NOMBRE'] == "MENDOZA AERO"]

# Ordenar por FECHA (de más antiguo a más reciente)
df_mendoza_aero = df_mendoza_aero.sort_values(by='FECHA')

# Guardar archivo con solo "MENDOZA AERO"
df_mendoza_aero.to_parquet(
    'mendoza_aero.parquet',
    engine='pyarrow',
    compression='snappy'
)

print("Proceso completado. Archivos guardados:")
print("- datos_filtrados.parquet (todos los datos filtrados)")
print("- mendoza_aero.parquet (solo MENDOZA AERO)")

# Leer los archivos Parquet
df_filtrados = pd.read_parquet('datos_filtrados.parquet')
df_mendoza = pd.read_parquet('mendoza_aero.parquet')

print("Datos filtrados:")
print(df_filtrados.head())

print("\nDatos de MENDOZA AERO:")
print(df_mendoza.head())