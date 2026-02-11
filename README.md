# SMS Marketing Automation System (RFM-Based)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)
![PostgreSQL](https://img.shields.io/badge/postgresql-15+-336791.svg)

A modular, high-performance marketing automation system. This application enables segment-based customer targeting, AI-powered product recommendations, and automated Urdu SMS orchestration with integrated link tracking.

---

## 🚀 Key Features

### 🎯 Intelligent Segmentation
- **RFM Analysis Integration**: Processes customer data based on Recency, Frequency, and Monetary values.
- **Strategy-Driven**: Maps customer segments to specific marketing strategies defined in `Strategy.xlsx`.

### 🧠 Hybrid Recommendation Engine
- **Content-Based Filtering**: Recommends products similar to a customer's purchase history using TF-IDF and Cosine Similarity.
- **Collaborative Filtering**: Leverages an XGBoost model to predict products a customer is likely to purchase next.
- **Diversity Control**: Ensures a varied mix of recommendations to prevent customer fatigue.

### ✍️ Automated LLM-Powered Localization
- **Urdu SMS Generation**: Automatically generates natural, spoken-style Urdu SMS templates using DeepSeek-V3 via Ollama.
- **Dynamic Personalization**: Injects customer names and recommended products into templates in real-time.

### ⚡ Technical Optimizations
- **Asynchronous SMS Orchestration**: Utilizes `aiohttp` and `asyncio` for high-throughput, non-blocking SMS transmission.
- **Smart Caching**: Implements Streamlit `@st.cache_resource` for database persistence and ML model efficiency.
- **Link Tracking**: Automatically generates and tracks shortlinks for every customer to monitor campaign conversion.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.10
- **Database**: PostgreSQL (psycopg2)
- **ML/AI**: Scikit-learn, XGBoost, DeepSeek-V3 (via LLM API)
- **Asynchronous I/O**: aiohttp, asyncio

---

## 📋 Prerequisites

- Python 3.10 or higher
- PostgreSQL instance
- Ollama (running DeepSeek-V3) or compatible LLM API
- SMS Gateway API access

---

## 🔧 Installation & Setup

Please refer to the [Setup Guide](TECHNICAL_WALKTHROUGH.md#installation--setup) in the technical documentation for detailed instructions on:
1. Creating a dedicated Anaconda environment.
2. Configuring the `.env` file with your credentials.
3. Installing dependencies via `requirements.txt`.

---

## 📖 Usage Flow

1. **Upload**: Provide a customer list in Excel format.
2. **Fetch History**: The system retrieves purchase data and generates hybrid AI recommendations.
3. **Generate Content**: LLM creates localized Urdu templates based on segment-specific strategies.
4. **Personalize**: Templates are personalized for each user with their specific name and product list.
5. **Send**: SMS are dispatched via the asynchronous gateway with integrated tracking links.

---

## 📄 License & Contact

This project is proprietary and confidential. For support or contributions, please contact the development team.
