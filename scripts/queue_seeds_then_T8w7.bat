@echo off
rem Runs, one after another (each keeps Windows awake while it runs, see run_full.py):
rem   1. seed 1 and seed 2 of avg + beltpulse (3 views)  -> data\models\full_s1, full_s2   (~1 h)
rem   2. the full 8-view run (T8w7), all 6 models          -> data\models\full_T8w7        (~4-6 h)
rem Progress: outputs\logs\queue_status.txt and the per-run logs in outputs\logs\
cd /d "%~dp0.."
set PY=.venv\Scripts\python
echo %date% %time%  queue started> outputs\logs\queue_status.txt
%PY% scripts\run_full.py --seed 1 --modes avg beltpulse --from 06
echo %date% %time%  seed 1 finished (exit %errorlevel%)>> outputs\logs\queue_status.txt
%PY% scripts\run_full.py --seed 2 --modes avg beltpulse --from 06
echo %date% %time%  seed 2 finished (exit %errorlevel%)>> outputs\logs\queue_status.txt
%PY% scripts\run_full.py --variant T8w7
echo %date% %time%  T8w7 finished (exit %errorlevel%)>> outputs\logs\queue_status.txt
echo %date% %time%  QUEUE DONE>> outputs\logs\queue_status.txt
