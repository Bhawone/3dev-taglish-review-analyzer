## Streamlit Prototype

Lightweight interface for demoing the Taglish Review Analyzer.

### Quick start

1. Install dependencies (Streamlit + requests already covered if you followed the API instructions):
   ```
   pip install streamlit requests
   ```
2. Launch the app:
   ```
   streamlit run app/ui/app.py
   ```
3. Optional: point the UI at the FastAPI backend by setting the environment variable before launching:
   ```
   set REVIEW_API_URL=http://127.0.0.1:8000
   ```
   or on PowerShell:
   ```
   $env:REVIEW_API_URL="http://127.0.0.1:8000"
   ```

By default, the app imports `analyze_text` directly for zero-setup experimentation.


