@echo off
REM Setup script for Tienet Model
REM Activates virtual environment and installs requirements

echo ========================================
echo Tienet Model Environment Setup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "myenv\Scripts\activate.bat" (
    echo Error: Virtual environment 'myenv' not found!
    echo Please create it first with: python -m venv myenv
    pause
    exit /b 1
)

echo Activating virtual environment...
call myenv\Scripts\activate.bat

echo.
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True)"

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the environment manually, run:
echo   myenv\Scripts\activate
echo.
pause

