@echo off
echo pavlovmusic setup

echo looking
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo python not installed or in path
    echo go install python dumbass
    pause
    exit /b 1
)

echo found it!!!!!!!!

echo depnendi cieds
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo error: failed
    pause
    exit /b 1
)

echo installed

python pavlovmusic.py

