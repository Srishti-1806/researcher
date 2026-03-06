
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

cp .env.example .env   # only if .env missing
# then edit .env to set GROQ_API_KEY, GROQ_MODEL_NAME, TAVILY_API_KEY, etc.

python verify_setup.py

python main.py

streamlit run app.py
