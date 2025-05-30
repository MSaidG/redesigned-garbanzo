@echo off

:: Activate the virtual environment
call "myenv\Scripts\activate"

:: Check if activation was successful
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b
)
echo Virtual environment activated successfully


echo Running Streamlit app...
:: Run your Streamlit application
streamlit run app.py

:: Keep the window open to see any errors
echo.
echo Press any key to close this window...
pause >nul