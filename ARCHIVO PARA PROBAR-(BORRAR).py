import pandas as pd

# Ruta al archivo Parquet
ruta_archivo = 'Tablas/SANCIONES_30_31_x_y.parquet'

# Leer el archivo Parquet
df = pd.read_parquet(ruta_archivo)

# Mostrar las primeras 5 filas
print(df.head())