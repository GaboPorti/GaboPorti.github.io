@echo on
setlocal
chcp 65001 >NUL

REM === Ruta del proyecto ===
cd /d "C:\Users\gabri\OneDrive\Escritorio\mi_proyecto_kpis" || (echo [ERROR] Carpeta no existe & pause & exit /b)

REM === Archivo de la app (cambia si lo tuyo se llama distinto) ===
set "APP=app.py"

REM === Elegir un Python para crear el venv si falta (py -> Anaconda -> python) ===
set "PYEXE="
where py >NUL 2>&1 && set "PYEXE=py -3"
if not defined PYEXE if exist "%USERPROFILE%\anaconda3\python.exe" set "PYEXE=%USERPROFILE%\anaconda3\python.exe"
if not defined PYEXE if exist "%USERPROFILE%\miniconda3\python.exe" set "PYEXE=%USERPROFILE%\miniconda3\python.exe"
if not defined PYEXE where python >NUL 2>&1 && set "PYEXE=python"

REM === Crear venv si no existe ===
if not exist "venv\Scripts\python.exe" (
  if not defined PYEXE (
    echo [ERROR] No se encontro ningun Python para crear el entorno. Abre este .bat desde **Anaconda Prompt** o instala Python.
    pause & exit /b
  )
  echo [INFO] Creando entorno virtual...
  %PYEXE% -m venv venv || (echo [ERROR] No pude crear venv & pause & exit /b)
)

REM === Usar SIEMPRE el python del venv (sin activar)
set "VENV_PY=%cd%\venv\Scripts\python.exe"
if not exist "%VENV_PY%" (echo [ERROR] Falta venv\Scripts\python.exe & pause & exit /b)

"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install streamlit==1.37.0 pandas==2.2.2 numpy==1.26.4 || (echo [ERROR] Fallo instalacion & pause & exit /b)

REM === Ejecutar la app con el python del venv (sin activar)
"%VENV_PY%" -m streamlit run "%APP%"

pause
