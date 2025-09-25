@echo on
setlocal
chcp 65001 >NUL

REM === Ruta del proyecto (AJUSTADA A TU ONEDRIVE) ===
cd /d "C:\Users\gabri\OneDrive\Escritorio\mi_proyecto_kpis" || (
  echo [ERROR] No existe la carpeta del proyecto
  pause & exit /b
)

REM === Nombre de tu archivo de app (usa el que tienes ahora) ===
set "APP=hospital_kpis_app_mvp_streamlit_python.py"

if not exist "%APP%" (
  echo [ERROR] No encuentro %APP% en %cd%
  dir
  pause & exit /b
)

REM === Buscar un Python utilizable (py -> Anaconda -> python) ===
set "PY="
where py >NUL 2>&1 && set "PY=py"

if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"

if not defined PY (
  where python >NUL 2>&1 && set "PY=python"
)

if not defined PY (
  echo [ERROR] No se encontro Python en el sistema.
  echo - Sugerencia: abre este .bat desde "Anaconda Prompt" o instala Python.
  pause & exit /b
)

REM === Crear venv si no existe (con el Python detectado) ===
if not exist "venv\Scripts\activate" (
  echo [INFO] Creando entorno virtual con: %PY%
  "%PY%" -m venv venv || (echo [ERROR] No pude crear venv & pause & exit /b)
)

REM === Activar venv ===
call "venv\Scripts\activate" || (echo [ERROR] No pude activar venv & pause & exit /b)

REM === Instalar deps usando el Python DEL VENV (evita alias del Store) ===
python -m pip install --upgrade pip
python -m pip install streamlit==1.37.0 pandas==2.2.2 numpy==1.26.4 || (
  echo [ERROR] Fallo instalacion de dependencias
  pause & exit /b
)

REM === Lanzar Streamlit (quedara la consola abierta) ===
streamlit run "%APP%"
pause
