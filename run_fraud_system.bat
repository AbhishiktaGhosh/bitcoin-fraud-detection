@echo off
echo ============================================================
echo Starting GraphGuard Fraud Detection Terminal (Subdirectory)
echo ============================================================
cd fraud-system
call venv\Scripts\activate

echo Launching FastAPI Backend in the background...
start /B uvicorn api.main:app --host 127.0.0.1 --port 8000

echo Launching Streamlit Dashboard...
python -m streamlit run dashboard/app.py

pause
