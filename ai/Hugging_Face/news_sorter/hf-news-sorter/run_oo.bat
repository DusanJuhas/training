@echo off
rem Runs the object-oriented implementation of the News Sorter inside the hf-env virtual environment.
rem Any arguments are passed on to run_all.py, e.g.:  run_oo.bat --with-finetune
setlocal

rem %~dp0 = folder of this batch file, so it works from any current directory.
call "%~dp0hf-env\Scripts\activate.bat"
if errorlevel 1 exit /b 1

rem run_all.py is started from its own folder.
pushd "%~dp0oo"
python run_all.py %*
set "EXITCODE=%ERRORLEVEL%"
popd

exit /b %EXITCODE%
