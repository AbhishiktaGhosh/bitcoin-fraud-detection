@echo off
echo ============================================================
echo Starting Original GNN Dashboard (fraud-gnn-dashboard)
echo ============================================================
cd fraud-gnn-dashboard
call ..\fraud-system\venv\Scripts\activate

echo Launching Streamlit Dashboard...
python -m streamlit run app.py

pause
