![result](https://github.com/user-attachments/assets/e547da21-0387-4289-8ec9-e76ce32e4387)

🛡️ LLM Evaluation: Automated Quality Assurance for Chatbots

An **open-source**, **local-first** system for detecting hallucinations and relevance issues in AI responses—**in real-time**.

---

## 📖 Overview

This project is designed to **automate the evaluation of AI chatbots** by comparing **user queries**, **retrieved context**, and **AI responses**. It uses **local NLP models** (no expensive APIs!) to generate two key metrics:

- **Relevance Score**: Did the AI actually answer the user’s question?
- **Faithfulness Score**: Is the AI’s response grounded in the provided context, or is it making things up?

---

## 🏗️ Architecture & Engineering Decisions

### The Challenge: Scalability & Cost
The system is built to handle **millions of daily conversations** with **minimal latency and zero variable cost**.

#### 1. **The "Zero-Cost" Decision**
Instead of relying on OpenAI’s API (which costs ~$5.00 per 1M tokens), we use **local models**:
- **Semantic Relevance**: `all-MiniLM-L6-v2` (Bi-Encoder) – **fast (~10ms inference)** and lightweight.
- **Hallucination Detection**: `cross-encoder/nli-distilroberta-base` – performs **Natural Language Inference (NLI)** to check if the context logically supports the response.

**Result**: The pipeline runs **entirely offline**, with **$0.00 variable cost** and **full data privacy**.

#### 2. **Robust Data Loading**
The input data was messy—**invalid JSON syntax, mismatched records, and inconsistent formats**. We built a **custom `loader.py`** with:
- **Regex-based sanitization** to fix broken JSON.
- **Fuzzy matching** to link context vectors to the correct chat turn.
- **Fallback logic** to extract missing responses from metadata.

---

## ⚠️ The Data Challenge (And How We Fixed It)

The provided JSON datasets had **syntax errors, structural inconsistencies, and missing data**:
- **Invalid JSON**: C-style comments (`// ...`) and trailing commas.
- **Data Mismatches**: In some cases, the chat history ended before the AI’s final answer.
- **Format Inconsistencies**: AI responses stored as strings, lists, or even missing entirely.

### **The Solution (`src/loader.py`)**
- **Auto-repairs JSON**: Strips comments and trailing commas before parsing.
- **Smart Matching**: Uses fuzzy logic to pair context with the right chat turn.
- **Fallback Extraction**: If the answer is missing from the chat log, it pulls it from the vector metadata.

---

## 🚀 Installation & Usage

### 1. **Clone the Repository**
```
git clone <your-repo-url>
cd llm-evaluation
```

### 2. **Install Dependencies**
*(Recommended: Use a virtual environment.)*
```
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install libraries
pip install -r requirements.txt
```

### 3. **Set Up Data**
Place your JSON files in the `data/` directory:
```
data/
├── chat_history.json
└── context_data.json
```
*(The pipeline auto-repairs JSON, so raw files work fine.)*

### 4. **Run the Dashboard**
We use **Streamlit** for an interactive report:
```
streamlit run dashboard.py
```

---

## 📊 Evaluation Logic

Once the dashboard is running:
1. Click **"Run Evaluation"**.
2. The system calculates metrics **locally**:
   - **Pass**: Score ≥ Threshold (default: **0.7**).
   - **Fail**: Score < Threshold.

### **Example (Case 01)**
- **Context**: *"Gopal Mansion rooms cost Rs 800."*
- **AI Response**: *"We offer rooms at our clinic for Rs 2000."*
- **Result**: The NLI model detects a **contradiction**.
- **Faithfulness Score**: **~0.56 (Low Confidence)**
- **Status**: **🚨 HALLUCINATION DETECTED** *(if threshold = 0.7)*

---

## 📂 Project Structure
```
llm-evaluation/
│
├── data/               # Input JSON files
│   ├── chat_history.json
│   └── context_data.json
│
├── src/                # Core logic
│   ├── loader.py       # Robust JSON parser & matcher
│   └── metrics.py      # Local NLP models (MiniLM + NLI)
│
├── dashboard.py        # Streamlit UI entry point
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🤝 Contributing
We welcome contributions! Open an **issue** or submit a **pull request** for:
- New metric implementations.
- Performance optimizations.
- Bug fixes.

