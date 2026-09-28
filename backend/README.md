python -m venv venv

.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install pandas numpy matplotlib seaborn scikit-learn notebook joblib

python -m pip install -r requirements.txt

to verify
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, joblib; print('Everything installed successfully')"


for fast api
python -m pip install fastapi "uvicorn[standard]"
uvicorn api.main:app --reload

python create_indexes.py
python -m uvicorn api.main:app --reload