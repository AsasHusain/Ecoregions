@echo off
rem Local globe viewer: serves site\ on http://localhost:8000 and opens it in the browser.
rem Needs only Python's standard library. Close this window to stop the server.
curl -s -o nul http://localhost:8000/ && (start "" http://localhost:8000/ & exit /b)
cd /d "%~dp0..\site"
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8000/"
where py >nul 2>nul && (py -m http.server 8000) || (python -m http.server 8000)
