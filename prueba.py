import pandas as pd
import os
import chardet

# Ruta del archivo
seguimiento_path = os.path.join('Tablas', 'seguimiento_exptes_dpl.xls')

# Intentar leer el archivo con diferentes codificaciones
codificaciones = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']

for encoding in codificaciones:
    try:
        print(f"Intentando leer el archivo con codificación: {encoding}")
        df_seguimiento = pd.read_csv(seguimiento_path, delimiter='\t', encoding=encoding)
        print("Archivo leído con éxito!")
        print(df_seguimiento.head())  # Muestra las primeras filas del DataFrame
        break  # Salir del bucle si se lee con éxito
    except Exception as e:
        print(f"Error al leer con codificación {encoding}: {e}")

        

# Verificar la codificación del archivo
with open(seguimiento_path, 'rb') as f:
    rawdata = f.read()
    result = chardet.detect(rawdata)
    encoding_detected = result['encoding']
    print(f"Codificación detectada: {encoding_detected}")