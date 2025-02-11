import requests
from datetime import datetime, timedelta
import os

# Crear la carpeta "Datos_meteorológicos" en la misma ubicación del script
ruta_carpeta = "Datos_meteorológicos"
if not os.path.exists(ruta_carpeta):
    os.makedirs(ruta_carpeta)
    print(f"Carpeta creada: {ruta_carpeta}")

# Función para descargar el archivo con un encabezado User-Agent
def descargar_archivo(fecha):
    base_url = "https://ssl.smn.gob.ar/dpd/descarga_opendata.php?file=observaciones/datohorario"
    fecha_str = fecha.strftime("%Y%m%d")
    url = f"{base_url}{fecha_str}.txt"
    archivo_nombre = f"datohorario{fecha_str}.txt"
    archivo_path = os.path.join(ruta_carpeta, archivo_nombre)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lanza una excepción si hay un error en la descarga
        with open(archivo_path, "wb") as file:
            file.write(response.content)
        print(f"Archivo descargado: {archivo_nombre}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar {url}: {e}")

# Fecha inicial y fecha final
fecha_inicio = datetime(2024, 1, 1)
fecha_final = datetime(2025, 2, 8)

# Iterar sobre todas las fechas y descargar los archivos correspondientes
fecha_actual = fecha_inicio
while fecha_actual <= fecha_final:
    descargar_archivo(fecha_actual)
    fecha_actual += timedelta(days=1)

print("Descargas finalizadas.")
