# Technical Walkthrough: SMS Marketing Automation System (RFM-Based)

This document provides a deep dive into the architecture, module design, and technical workflows of the BaKhabar SMS Recommendation System.

---

## 🏗️ System Architecture

The application follows a modular architecture designed for scalability, performance, and security. It separates concerns into specialized modules for database management, recommendation logic, LLM interaction, and SMS orchestration.

### Core Modules

| Module | Responsibility |
| :--- | :--- |
| `app.py` | The main Streamlit entry point; handles UI state and orchestration. |
| `config.py` | Centralized configuration using `python-dotenv` for secure environment management. |
| `recommendations.py` | The AI engine; implements hybrid, content-based, and collaborative filtering. |
| `templates.py` | Orchestrates LLM prompts to generate localized Urdu SMS templates. |
| `llm.py` | Handles robust communication with the LLM API including retries and error handling. |
| `sms.py` | Optimized asynchronous SMS delivery gateway using `aiohttp`. |
| `db.py` | Manages PostgreSQL persistence and shortlink generation logic. |
| `utils.py` | Shared utility functions for phone normalization and data formatting. |

---

## 🧠 Recommendation Engine (`recommendations.py`)

The system employs a **Hybrid Recommendation** strategy to maximize accuracy and diversity.

### 1. Content-Based Filtering
- **Logic**: Analyzes product metadata (name, tags, description).
- **Tech**: Uses `TfidfVectorizer` to convert text to vectors and `cosine_similarity` to find similar items.
- **Goal**: Recommend items similar to what the user has purchased before.

### 2. Collaborative Filtering
- **Logic**: Predicts user preferences based on broader community patterns.
- **Tech**: Utilizes a pre-trained **XGBoost** model (`xgb_model.pkl`) to score "User-Product" pairs based on likelihood of purchase.

### 3. Diversity & Reranking
- The engine uses a custom diversification algorithm to ensure that the final 5 recommendations aren't too structurally similar, preventing redundant suggestions.

---

## 🔗 Link Tracking & Shortlinks (`db.py`)

Every SMS sent contains a personalized tracking link.
- When a message is generated, the system creates a unique hash for that customer.
- It stores the original destination URL and the `short_code` in the `sms_link_tracking` table.
- This allows for fine-grained analytics on Click-Through Rates (CTR) per campaign and per segment.

---

## ⚡ Asynchronous SMS Gateway (`sms.py`)

To handle large-scale campaigns, the system avoids synchronous blocking requests.
- **Asyncio Loop**: The `send_sms_batch` function initiates an asynchronous loop.
- **Concurrency Control**: A `Semaphore` is used to limit concurrent connections to 50, preventing API rate-limiting or network congestion.
- **Persistence**: Results (Success/Failure) are streamed into the `sms_results` table for audit logging.

---

## ⚙️ Installation & Setup

### 1. Environment Configuration
Create a `.env` file in the root directory (refer to `.env.example`):
```env
PG_HOST=your_host
PG_USER=your_user
PG_PASSWORD=your_password
SMS_AUTH_HEADER=Basic your_header
```

### 2. Anaconda Environment Setup
```bash
# Create environment
conda create -n bakhabar_env python=3.10 -y

# Activate
conda activate bakhabar_env

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App
```bash
streamlit run app.py
```

---

## 🛡️ Best Practices Implemented

1. **Security**: Zero hardcoded secrets; all sensitive data is injected via environment variables.
2. **Efficiency**: Streamlit `cache_resource` is used for DB connections and ML models to prevent repetitive I/O.
3. **Robustness**: The LLM module includes exponential backoff retries for API resilience.
4. **Maintenance**: Proper `.gitignore` ensures that transient logs and sensitive `.env` files are never committed to version control.
