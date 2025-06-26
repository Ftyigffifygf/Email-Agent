# Email Agent with LangChain and RAG

An intelligent email assistant that uses Retrieval-Augmented Generation (RAG) to provide contextual responses based on your email history and knowledge base.

## Features

- 📧 **Email Classification**: Automatically categorize and prioritize emails
- 🔍 **Smart Search**: Find relevant emails using semantic search
- ✍️ **Response Generation**: Generate contextual email replies using RAG
- 📊 **Email Summarization**: Summarize email threads and conversations
- 🤖 **AI Agent**: Conversational interface for email management
- 📱 **Gmail Integration**: Connect with Gmail API for seamless access
- 🧠 **Memory**: Maintains conversation context for better interactions

## Architecture

The system combines several powerful technologies:

- **LangChain**: For building the AI agent and chains
- **OpenAI GPT**: For natural language understanding and generation
- **FAISS**: Vector database for semantic search
- **Gmail API**: For email access and management
- **RAG Pipeline**: Retrieval-Augmented Generation for contextual responses

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/email-agent-rag.git
cd email-agent-rag
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### 1. OpenAI API Key

Get your API key from [OpenAI](https://platform.openai.com/api-keys) and add it to your `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Gmail API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Gmail API
4. Create credentials (OAuth 2.0 Client ID)
5. Download the credentials JSON file and save as `credentials.json`

Required scopes:
- `https://www.googleapis.com/auth/gmail.readonly`
- `https://www.googleapis.com/auth/gmail.send`

## Usage

### Basic Usage

```python
from email_agent import EmailAgent

# Initialize the agent
agent = EmailAgent(
    openai_api_key="your-api-key",
    gmail_credentials="credentials.json"
)

# Sync recent emails
agent.sync_emails(max_results=100)

# Process queries
response = agent.process_query("What are the most important emails this week?")
print(response)

# Generate email reply
reply = agent.draft_email(
    recipient="colleague@company.com",
    subject="Re: Project Update",
    context="Need to follow up on the quarterly review discussion"
)
print(reply)
```

### Command Line Interface

```bash
python email_agent.py
```

### Available Commands

The agent supports various natural language commands:

- **Search emails**: "Find emails about project deadlines"
- **Classify emails**: "What's the priority of this email?"
- **Generate replies**: "Help me respond to this email"
- **Summarize**: "Summarize emails from last week"
- **Draft emails**: "Draft an email to the team about the meeting"

## Project Structure

```
email-agent-rag/
├── email_agent.py          # Main application
├── requirements.txt        # Dependencies
├── README.md              # Documentation
├── .env.example           # Environment template
├── .gitignore            # Git ignore rules
├── credentials.json       # Gmail API credentials (not in repo)
├── token.pickle          # Gmail API token (auto-generated)
├── email_vectorstore/    # FAISS vector store (auto-generated)
├── tests/               # Unit tests
├── examples/            # Usage examples
└── docs/               # Additional documentation
```

## Data Flow

1. **Email Ingestion**: Fetch emails from Gmail API
2. **Preprocessing**: Clean and structure email data
3. **Embedding**: Convert emails to vector embeddings
4. **Storage**: Store in FAISS vector database
5. **Retrieval**: Search for relevant emails based on queries
6. **Generation**: Use LLM with retrieved context to generate responses

## API Reference

### EmailAgent Class

#### Methods

- `sync_emails(query="", max_results=100)`: Sync emails from Gmail
- `process_query(query)`: Process natural language queries
- `draft_email(recipient, subject, context)`: Draft new emails
- `get_email_insights()`: Get email analytics

### EmailVectorStore Class

#### Methods

- `add_emails(emails)`: Add emails to vector store
- `search_similar_emails(query, k=5)`: Search for similar emails
- `save_store()`: Persist vector store to disk

### GmailHandler Class

#### Methods

- `get_emails(query="", max_results=100)`: Retrieve emails
- `send_email(to, subject, body)`: Send emails
- `authenticate()`: Handle Gmail API authentication

## Configuration Options

### Environment Variables

```env
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
GMAIL_CREDENTIALS_FILE=credentials.json
TOKEN_FILE=token.pickle
VECTOR_STORE_PATH=email_vectorstore
LOG_LEVEL=INFO
```

### Agent Parameters

```python
agent = EmailAgent(
    openai_api_key="your-key",
    gmail_credentials="credentials.json",
    model="gpt-3.5-turbo",  # or gpt-4
    temperature=0.7,
    max_tokens=1000,
    chunk_size=1000,
    chunk_overlap=200
)
```

## Examples

### Email Search
```python
# Search for project-related emails
results = agent.process_query("Find all emails about the Q4 project review")
```

### Email Classification
```python
# Classify email priority
classification = agent.process_query("What's the priority of the email from John about the deadline?")
```

### Response Generation
```python
# Generate contextual reply
reply = agent.process_query("Help me write a professional response to the client inquiry about pricing")
```

## Security Considerations

- Store API keys securely using environment variables
- Never commit `credentials.json` or `token.pickle` to version control
- Use OAuth 2.0 for Gmail API authentication
- Implement proper error handling for API failures
- Consider rate limiting for API calls

## Troubleshooting

### Common Issues

1. **Gmail API Authentication Error**
   - Ensure `credentials.json` is properly configured
   - Check that Gmail API is enabled in Google Cloud Console
   - Verify OAuth consent screen is set up

2. **OpenAI API Errors**
   - Check API key validity
   - Monitor usage limits and quotas
   - Handle rate limiting in your application

3. **Vector Store Issues**
   - Ensure sufficient disk space for FAISS index
   - Check file permissions for vector store directory

4. **Memory Issues**
   - Reduce `max_results` when syncing emails
   - Implement batch processing for large email volumes

## Performance Optimization

- **Batch Processing**: Process emails in batches to avoid memory issues
- **Incremental Updates**: Only sync new emails since last update
- **Caching**: Cache frequently accessed embeddings
- **Filtering**: Pre-filter emails by date, sender, or keywords

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=email_agent

# Run specific test
pytest tests/test_email_agent.py::TestEmailAgent::test_sync_emails
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.com/) for the agent framework
- [OpenAI](https://openai.com/) for the language models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Gmail API](https://developers.google.com/gmail/api) for email access

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/email-agent-rag/issues) page
2. Review the documentation
3. Create a new issue with detailed information

## Roadmap

- [ ] Add support for other email providers (Outlook, Yahoo)
- [ ] Implement email templates and signatures
- [ ] Add calendar integration
- [ ] Support for attachments
- [ ] Web interface using Streamlit or FastAPI
- [ ] Mobile app integration
- [ ] Advanced analytics and reporting
- [ ] Multi-language support
