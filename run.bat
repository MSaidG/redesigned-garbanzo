@echo off
echo Creating Python environment "myenv"...
python -m venv myenv

:: Activate the virtual environment
call "myenv\Scripts\activate"

:: Check if activation was successful
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b
)
echo Virtual environment activated successfully


echo Installing pip upgrades...
python -m pip install --upgrade pip

echo Environment ready!

echo Installing packages from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo Failed to install some packages. Check requirements.txt.
) else (
    echo All packages installed successfully!
)


echo Running Streamlit app...
:: Run your Streamlit application
streamlit run app.py

:: Keep the window open to see any errors
echo.
echo Press any key to close this window...
pause >nul