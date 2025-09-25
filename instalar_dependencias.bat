@echo off
setlocal
chcp 65001 >NUL

REM === Configuración ===
set "PROJECT_DIR=C:\Users\gabri\OneDrive\Escritorio\mi_proyecto_kpis"
set "PYEXE=%PROJECT_DIR%\venv\Scripts\python.exe"

echo *** Entrando a la carpeta del proyecto ***
cd /d "%PROJECT_DIR%" || (echo [ERROR] Carpeta no encontrada & pause & exit /b)

REM Si no existe el intérprete del venv, intenta crearlo con el launcher "py"
if not exist "%PYEXE%" (
  echo *** No encuentro el entorno virtual. Intento crearlo... ***
  where py >NUL 2>&1 && (py -3 -m venv venv) || (echo [ERROR] No pude crear venv. Instala Python y el launcher "py". & pause & exit /b)
)

echo *** Instalando dependencias desde requirements.txt ***
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r requirements.txt

echo *** Listo. Dependencias instaladas. ***
pause
