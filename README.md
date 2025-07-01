# AI FUSION

**Multi-Model AI Intelligence System**

Combining specialized AI models for optimal task-specific solutions.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview
AI FUSION is a web-based platform that leverages multiple specialized AI models to provide optimal, task-specific solutions. It automatically detects the nature of a user's query and routes it through the best-suited AI models for:
- Code generation
- Creative writing
- Logical reasoning
- Mathematical computation
- Deep document understanding
- Summarization
- Web-integrated answers
- UI code generation (React + Tailwind)
- And more

## Features
- **Automatic Query Categorization:** Detects the type of user query and selects the best AI models.
- **Multi-Model Pipeline:** Combines outputs from several advanced models for refined, validated answers.
- **Web Interface:** User-friendly web UI for submitting queries and viewing multi-stage AI responses.
- **Extensible:** Easily add or modify AI models and categories.

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd AI_fussion
   ```
2. **Install dependencies:**
   Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   Then install the required packages:
   ```bash
   pip install flask requests scikit-learn pandas joblib groq pytube
   ```
3. **Prepare data and models:**
   - Ensure `actual_data.xlsx` is present in the root directory.
   - Run `model.py` once to train and save the model artifacts (`logistic_model.pkl`, `tfidf_vectorizer.pkl`, `reverse_label_mapping.pkl`).

4. **Run the application:**
   ```bash
   python app.py
   ```
   The app will be available at [http://localhost:5000](http://localhost:5000).

## Usage
- Open your browser and go to [http://localhost:5000](http://localhost:5000).
- Enter your question or task in the input box.
- The system will automatically select and run the best AI models for your query, displaying multi-stage responses.

## Requirements
- Python 3.8+
- Flask
- requests
- scikit-learn
- pandas
- joblib
- groq
- pytube (for YouTube audio download feature)

## Project Structure
```
AI_fussion/
├── app.py                # Main Flask app
├── app_flask.py          # (Alternative/legacy Flask app)
├── model.py              # Model training and saving
├── music.py              # YouTube audio download utility
├── sasta_array.py        # (Utility or experimental script)
├── index.html            # Web UI template
├── actual_data.xlsx      # Training data
├── logistic_model.pkl    # Trained model
├── tfidf_vectorizer.pkl  # Saved vectorizer
├── reverse_label_mapping.pkl # Label mapping
└── templates/            # (If used for Flask templates)
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. 