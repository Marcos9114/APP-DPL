import pandas as pd

# Ruta del archivo
file_path = r'Tablas\PLANILLA FACTIBILIDADES desde 2020 v2 (1).xlsx'

# Leer el archivo de Excel
df = pd.read_excel(file_path)

# Mostrar todas las columnas del archivo
print(df.columns)