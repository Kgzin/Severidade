@echo off
cd /d "%~dp0"
if not exist .venv (
    py -V:3.12 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
.venv\Scripts\streamlit.exe run app.py
