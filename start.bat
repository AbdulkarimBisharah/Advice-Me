@echo off
echo Starting Advisor Co-Pilot API...
echo.
echo Make sure ANTHROPIC_API_KEY is set:
echo   set ANTHROPIC_API_KEY=sk-ant-...
echo.
uvicorn backend_app:app --reload --port 8000
