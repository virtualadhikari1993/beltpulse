@echo off
rem Resumes queue_seeds_then_T8w7.bat after it stalled (seed 1 training was already done).
rem Start it hidden with output to a file, so a click in a console window cannot pause it (QuickEdit):
rem   Start-Process cmd -ArgumentList '/c','scripts\resume_queue.bat > outputs\logs\queue_console.txt 2>&1' -WindowStyle Hidden
cd /d "%~dp0.."
set PY=.venv\Scripts\python
echo %date% %time%  queue resumed (seed 1 from step 07)>> outputs\logs\queue_status.txt
%PY% scripts\run_full.py --seed 1 --modes avg beltpulse --from 07
echo %date% %time%  seed 1 finished (exit %errorlevel%)>> outputs\logs\queue_status.txt
%PY% scripts\run_full.py --seed 2 --modes avg beltpulse --from 06
echo %date% %time%  seed 2 finished (exit %errorlevel%)>> outputs\logs\queue_status.txt
%PY% scripts\run_full.py --variant T8w7
echo %date% %time%  T8w7 finished (exit %errorlevel%)>> outputs\logs\queue_status.txt
echo %date% %time%  QUEUE DONE>> outputs\logs\queue_status.txt
