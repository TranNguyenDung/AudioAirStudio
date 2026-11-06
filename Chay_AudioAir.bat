@echo off
rem Chay Audio Air Studio: uu tien ban .exe, neu chua co thi chay bang Python
cd /d "%~dp0"
set "EXE=dist\Audio Air Studio\Audio Air Studio.exe"
if exist "%EXE%" (
    start "" "%EXE%" %*
    exit /b
)
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "AudioAirStudio.py" %*
) else (
    python "AudioAirStudio.py" %*
)
