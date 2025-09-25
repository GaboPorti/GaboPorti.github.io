@echo off
setlocal
chcp 65001 >NUL

REM === Configuración ===
set "PROJECT_DIR=C:\Users\gabri\OneDrive\Escritorio\mi_proyecto_kpis"
set "PYEXE=%PROJECT_DIR%\venv\Scripts\python.exe"

echo *** Entrando a la carpeta del proyecto ***
cd /d "%PROJECT_DIR%" || (echo [ERROR] Carpeta no encontrada & pause & exit /b)

if not exist "%PYEXE%" (
  echo [ERROR] No encuentro el Python del venv en:
  echo   "%PYEXE%"
  echo Ejecuta primero: instalar_dependencias.bat
  pause
  exit /b
)

echo *** Lanzando aplicación con Streamlit ***
"%PYEXE%" -m streamlit run app.py

echo *** Proceso finalizado. ***
pause
