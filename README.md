# panem-et-promptes

This project implements a RAG pipeline for analyzing academic essays about *The Hunger Games* and evaluating the system’s retrieval and generation quality.

---

## 🧩 Setup & Installation

### Environment Setup
Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### API Key Setup
1. Get an OpenAI API key here: https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key" and copy the key that begins with `sk-....`
4. Duplicate the included `.env.example` file and rename it to `.env`
5. Paste your API key in the OPENAI_API_KEY field, like so:
    ```bash
    OPENAI_API_KEY=sk-your-key-here
    ```

### Running the Program
To run the interactive QA system:
```bash
python main.py
```

To run the evaluation script:
```bash
python evaluate.py
```

## 🧠 Approach & Design Decisions

## ⚙️ Familiar vs. New Technologies

## 📊 Evaluation Results & Interpretation

## ⚠️ Known Limitations

## 🚀 Future Improvements
