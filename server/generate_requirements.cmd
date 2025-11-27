@echo off
SETLOCAL

REM Activate Conda environment
echo Activating Conda environment 'pothole_detection'...
call conda activate pothole_detection
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate Conda environment.
    pause
    EXIT /B 1
) ELSE (
    echo [SUCCESS] Conda environment activated.
)

REM Generate requirements.txt
echo Generating requirements.txt...
pip freeze > requirements.txt
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to generate requirements.txt.
    pause
    EXIT /B 2
) ELSE (
    echo [SUCCESS] requirements.txt created successfully.
)

pause
