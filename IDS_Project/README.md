# AI-Based Intrusion Detection Agent (Academic Demo)

This project is a **simple, beginner-friendly** prototype of an AI-based intrusion detection system (IDS) with a Streamlit dashboard. It is intentionally lightweight and well-commented for university demos.

## Project Structure

```
IDS_Project/
├── data/sample.csv
├── model/lstm_model.h5
├── ai_agent.py
├── dashboard.py
└── alerts.log
```

## How the Dashboard Connects to the AI Agent

- `dashboard.py` loads the CSV data and **directly calls** the agent in `ai_agent.py`.
- The agent loads an LSTM model (`model/lstm_model.h5`) and predicts **NORMAL** or **ATTACK** for each traffic row.
- The dashboard processes **one row per refresh** to simulate near real-time detection.
- If an **ATTACK** is detected, the dashboard appends a message to `alerts.log`.

## Setup & Run

1. **Install dependencies** (TensorFlow is required to load the LSTM model):

```bash
pip install streamlit pandas numpy tensorflow
```

2. **Run the dashboard** from the `IDS_Project` directory:

```bash
cd IDS_Project
streamlit run dashboard.py
```

The browser will open automatically. Click **Start Monitoring** to begin.

## Notes

- If `model/lstm_model.h5` is missing or invalid, the agent will **create and save** a tiny LSTM model on first run (requires TensorFlow).
- This is a **prototype for academic demos**, not a production IDS.
