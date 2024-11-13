@echo off
set EXE_PATH=".\tts_server.exe"
set LOG_FILE=".\tts_server_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%-%TIME:~0,2%%TIME:~3,2%.txt"

echo Starting %EXE_PATH%...

%EXE_PATH% >> %LOG_FILE% 2>&1

echo Execution complete. Log file is located at %LOG_FILE%
