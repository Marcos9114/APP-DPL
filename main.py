import streamlit as st
from streamlit_gsheets import GSheetsConnection

try:
  # Assuming you have secrets configured
  gsheets_config = st.secrets["connections"]["gsheets"]

  # Create the connection using the service account credentials
  conn = GSheetsConnection(
      service_account_info={
          "project_id": gsheets_config["project_id"],
          "private_key": gsheets_config["private_key"],
          "client_email": gsheets_config["client_email"],
          "client_id": gsheets_config["client_id"],
      }
  )

  # ... rest of your code to read data from spreadsheets
except Exception as e:
  st.error(f"Error al leer los datos de las hojas de Google Sheets: {e}")