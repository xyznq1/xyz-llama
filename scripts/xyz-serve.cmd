@echo off
rem xyz-serve.cmd [MODEL] -- Ternary Bonsai 2 27B + our xyz v1.2 drafter: 160k context in about 8 GB of GPU memory, OpenAI API on :8080.
rem No MODEL given and none in models\? The first start downloads it (Hugging Face, cached after that), and our drafter too.
setlocal
set ROOT=%~dp0..
set MODEL=%~1
if "%MODEL%"=="" set MODEL=%ROOT%\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf
if "%XYZ_DRAFTER%"=="" set XYZ_DRAFTER=%ROOT%\models\xyz-v1.2-drafter.gguf
if "%XYZ_SERVER%"=="" set XYZ_SERVER=%ROOT%\build\bin\Release\llama-server.exe
set DRAFTER_URL=https://github.com/xyznq1/xyz-drafter/releases/download/v1.2/xyz-v1.2-drafter.gguf

set M=-m "%MODEL%"
if "%~1"=="" if not exist "%MODEL%" set M=-hf prism-ml/Ternary-Bonsai-2-27B-gguf -hff Ternary-Bonsai-2-27B-PTQ1_0.gguf
if exist "%XYZ_DRAFTER%" goto run
echo Downloading our drafter from %DRAFTER_URL%
curl.exe -fL --create-dirs -o "%XYZ_DRAFTER%.part" "%DRAFTER_URL%" && move /y "%XYZ_DRAFTER%.part" "%XYZ_DRAFTER%" >nul
if exist "%XYZ_DRAFTER%" goto run
echo No drafter: put xyz-v1.2-drafter.gguf from %DRAFTER_URL% in %ROOT%\models\
pause
exit /b 1

:run
"%XYZ_SERVER%" %M% -md "%XYZ_DRAFTER%" --no-mmproj -ngl 999 -c 163840 -fa on -ctk xyzkv2 -ctv xyzkv2 -ctkd q4_0 -ctvd q4_0 ^
  --spec-type draft-xyz --spec-draft-n-max 4 --spec-coupled --spec-rejection --parallel 1
rem started by a double-click? keep the window open so the error stays readable
if errorlevel 1 pause
