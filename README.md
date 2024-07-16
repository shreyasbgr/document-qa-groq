## Setting up the app
```conda create -p venv python=3.10 -y```

```conda activate ./venv```

```pip install -r requirements.txt```

While running locally, include a .env file with the variable with the below parameters
```
GOOGLE_API_KEY="<google-api-key>"
GROQ_API_KEY="<groq-api-key>" 
```

## Running the app

To run locally,
```streamlit run app-local.py```

To run on streamlit,
```streamlit run app-streamlit.py```