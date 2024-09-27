# GPT-powered advisor for resume critiques

## Features
- Critique resume content based on specific job description (using GPT-o1-preview)
- Critique resume visual design (using GPT-4o Vision Language)
- Revise resume based on critiques (using GPT-o1-preview)
  
## How to use

- Setup Environment
```
git clone https://github.com/benjaminlq/resume-editor
python3.10 -m venv venv
pip install --upgrade pip && pip install -r requirements.txt
pip install -e .
```

- Run Gradio Application
```
python /src/app.py

# gradio /src/app.py 
```