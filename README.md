# Mini Researcher

An autonomous web research assistant that searches, scrapes, summarizes, and compiles comprehensive research reports.

## Project Structure

```
mini-researcher/
│
├── backend/
│   ├── main.py         # API entry point (FastAPI server)
│   ├── researcher.py   # Research coordination agent
│   ├── search.py       # Search engine integration
│   ├── scraper.py      # Web page content extraction
│   ├── report.py       # Report generator and formatter
│   └── config.py       # App configuration and environment variables
│
├── frontend/
│   ├── index.html      # UI structure
│   ├── style.css       # Design & styling
│   └── script.js       # Client-side logic & API calls
│
├── .env                # Environment variables template
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Getting Started

### 1. Backend Setup
```bash
# Navigate to project directory
cd mini-researcher

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp .env .env.local  # or edit .env directly with your API keys

# Start backend server
uvicorn backend.main:app --reload
```

### 2. Frontend
Open `frontend/index.html` directly in your browser or serve it using a local static server.
