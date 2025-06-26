# Email Agent Configuration
# Copy this file to .env and fill in your actual values

# OpenAI API Key (Required)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Gmail API Configuration (Required)
GMAIL_CREDENTIALS_FILE=credentials.json
TOKEN_FILE=token.pickle

# Vector Store Configuration
VECTOR_STORE_PATH=email_vectorstore
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# LLM Configuration
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=1000

# Agent Configuration
MEMORY_WINDOW=5
MAX_EMAIL_SYNC=100

# Logging
LOG_LEVEL=INFO
LOG_FILE=email_agent.log

# Optional: Email Filtering
DEFAULT_EMAIL_QUERY=is:unread
EMAIL_CATEGORIES=work,personal,urgent,spam

# Optional: Performance Settings
MAX_SEARCH_RESULTS=5
BATCH_SIZE=50
ENABLE_CACHING=true
