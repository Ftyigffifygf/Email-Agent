"""
Email Agent with LangChain and RAG
==================================

An intelligent email assistant that uses Retrieval-Augmented Generation (RAG) 
to provide contextual responses based on your email history and knowledge base.

Features:
- Email classification and prioritization
- Contextual response generation using RAG
- Email summarization
- Automated email drafting
- Integration with Gmail API
- Vector database for email embeddings

Author: Girish G Shankar
License: This project is licensed under the terms of the MIT license.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.callbacks import get_openai_callback

# Email handling
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header

# Google API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Additional utilities
import pickle
import base64
from dataclasses import dataclass
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmailMessage:
    """Data class for email messages"""
    subject: str
    sender: str
    recipient: str
    body: str
    date: datetime
    message_id: str
    priority: str = "normal"
    category: str = "general"


class EmailVectorStore:
    """Manages vector storage for email content using FAISS"""
    
    def __init__(self, embeddings_model: OpenAIEmbeddings, store_path: str = "email_vectorstore"):
        self.embeddings = embeddings_model
        self.store_path = store_path
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_or_create_store(self):
        """Load existing vectorstore or create new one"""
        if os.path.exists(self.store_path):
            self.vectorstore = FAISS.load_local(self.store_path, self.embeddings)
            logger.info("Loaded existing vector store")
        else:
            # Create empty vectorstore
            docs = [Document(page_content="Initial document", metadata={"source": "init"})]
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            logger.info("Created new vector store")
    
    def add_emails(self, emails: List[EmailMessage]):
        """Add emails to vector store"""
        documents = []
        for email_msg in emails:
            # Create document content
            content = f"Subject: {email_msg.subject}\n"
            content += f"From: {email_msg.sender}\n"
            content += f"Date: {email_msg.date}\n"
            content += f"Body: {email_msg.body}"
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": email_msg.message_id,
                        "subject": email_msg.subject,
                        "sender": email_msg.sender,
                        "date": email_msg.date.isoformat(),
                        "chunk_id": i,
                        "category": email_msg.category,
                        "priority": email_msg.priority
                    }
                )
                documents.append(doc)
        
        if documents:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vectorstore.add_documents(documents)
            
            self.save_store()
            logger.info(f"Added {len(documents)} document chunks to vector store")
    
    def save_store(self):
        """Save vector store to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(self.store_path)
    
    def search_similar_emails(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar emails"""
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=k)
        return []


class GmailHandler:
    """Handles Gmail API interactions"""
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
              'https://www.googleapis.com/auth/gmail.send']
    
    def __init__(self, credentials_file: str = "credentials.json", token_file: str = "token.pickle"):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail API authenticated successfully")
    
    def get_emails(self, query: str = "", max_results: int = 100) -> List[EmailMessage]:
        """Retrieve emails from Gmail"""
        try:
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_results).execute()
            messages = results.get('messages', [])
            
            emails = []
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me', id=message['id']).execute()
                email_data = self._parse_email(msg)
                if email_data:
                    emails.append(email_data)
            
            logger.info(f"Retrieved {len(emails)} emails")
            return emails
            
        except Exception as e:
            logger.error(f"Error retrieving emails: {e}")
            return []
    
    def _parse_email(self, msg: Dict) -> Optional[EmailMessage]:
        """Parse Gmail API message into EmailMessage"""
        try:
            headers = msg['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
            to = next((h['value'] for h in headers if h['name'] == 'To'), '')
            date_str = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            # Parse date
            date = datetime.now()  # Fallback
            try:
                date = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
            except:
                pass
            
            # Extract body
            body = self._extract_body(msg['payload'])
            
            return EmailMessage(
                subject=subject,
                sender=sender,
                recipient=to,
                body=body,
                date=date,
                message_id=msg['id']
            )
            
        except Exception as e:
            logger.error(f"Error parsing email: {e}")
            return None
    
    def _extract_body(self, payload: Dict) -> str:
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body += base64.urlsafe_b64decode(data).decode('utf-8')
        elif payload['mimeType'] == 'text/plain':
            data = payload['body']['data']
            body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        return body
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email via Gmail API"""
        try:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            self.service.users().messages().send(
                userId='me', body={'raw': raw_message}).execute()
            
            logger.info(f"Email sent to {to}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


class EmailAgent:
    """Main email agent class with RAG capabilities"""
    
    def __init__(self, openai_api_key: str, gmail_credentials: str = "credentials.json"):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize components
        self.llm = OpenAI(temperature=0.7, max_tokens=1000)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = EmailVectorStore(self.embeddings)
        self.gmail_handler = GmailHandler(gmail_credentials)
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        
        # Load vector store
        self.vector_store.load_or_create_store()
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Initialize tools and agent
        self._setup_agent()
        
        logger.info("Email agent initialized successfully")
    
    def _setup_agent(self):
        """Setup LangChain agent with tools"""
        tools = [
            Tool(
                name="Search Emails",
                func=self._search_emails_tool,
                description="Search through email history for relevant information. Input should be a search query."
            ),
            Tool(
                name="Classify Email",
                func=self._classify_email_tool,
                description="Classify an email's priority and category. Input should be email content."
            ),
            Tool(
                name="Generate Reply",
                func=self._generate_reply_tool,
                description="Generate a professional email reply. Input should be the original email content."
            ),
            Tool(
                name="Summarize Emails",
                func=self._summarize_emails_tool,
                description="Summarize a collection of emails. Input should be email content or search query."
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
    
    def _search_emails_tool(self, query: str) -> str:
        """Tool for searching emails"""
        docs = self.vector_store.search_similar_emails(query, k=5)
        if docs:
            results = []
            for doc in docs:
                results.append(f"Email from {doc.metadata.get('sender', 'Unknown')}: {doc.page_content[:200]}...")
            return "\n\n".join(results)
        return "No relevant emails found."
    
    def _classify_email_tool(self, email_content: str) -> str:
        """Tool for classifying emails"""
        prompt = f"""
        Classify the following email by priority (high/medium/low) and category (work/personal/spam/urgent/information):
        
        Email: {email_content}
        
        Return format: Priority: [priority], Category: [category], Reason: [brief explanation]
        """
        return self.llm(prompt)
    
    def _generate_reply_tool(self, email_content: str) -> str:
        """Tool for generating email replies"""
        # Search for similar emails for context
        similar_docs = self.vector_store.search_similar_emails(email_content, k=3)
        context = "\n".join([doc.page_content for doc in similar_docs[:2]])
        
        prompt = f"""
        Generate a professional email reply based on the following original email and context from similar emails:
        
        Original Email: {email_content}
        
        Context from similar emails: {context}
        
        Generate a helpful, professional, and contextually appropriate reply:
        """
        return self.llm(prompt)
    
    def _summarize_emails_tool(self, query: str) -> str:
        """Tool for summarizing emails"""
        docs = self.vector_store.search_similar_emails(query, k=10)
        if docs:
            content = "\n".join([doc.page_content for doc in docs])
            prompt = f"""
            Summarize the following email content, highlighting key topics, decisions, and action items:
            
            {content[:3000]}  # Limit content to avoid token limits
            
            Provide a concise summary with bullet points for:
            - Key topics discussed
            - Important decisions made
            - Action items or follow-ups needed
            """
            return self.llm(prompt)
        return "No emails found to summarize."
    
    def sync_emails(self, query: str = "", max_results: int = 100):
        """Sync emails from Gmail and add to vector store"""
        logger.info("Syncing emails from Gmail...")
        emails = self.gmail_handler.get_emails(query, max_results)
        
        if emails:
            # Classify emails before adding to vector store
            for email in emails:
                classification = self._classify_email_tool(f"{email.subject}\n{email.body}")
                # Parse classification (simplified)
                if "high" in classification.lower():
                    email.priority = "high"
                elif "medium" in classification.lower():
                    email.priority = "medium"
                else:
                    email.priority = "low"
                
                if "work" in classification.lower():
                    email.category = "work"
                elif "personal" in classification.lower():
                    email.category = "personal"
                elif "urgent" in classification.lower():
                    email.category = "urgent"
            
            self.vector_store.add_emails(emails)
            logger.info(f"Synced {len(emails)} emails to vector store")
        else:
            logger.info("No new emails to sync")
    
    def process_query(self, query: str) -> str:
        """Process user query using the agent"""
        try:
            with get_openai_callback() as cb:
                response = self.agent.run(query)
                logger.info(f"Query processed. Tokens used: {cb.total_tokens}")
                return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def draft_email(self, recipient: str, subject: str, context: str) -> str:
        """Draft a new email with RAG context"""
        # Search for relevant context
        similar_docs = self.vector_store.search_similar_emails(context, k=3)
        context_info = "\n".join([doc.page_content for doc in similar_docs[:2]])
        
        prompt = f"""
        Draft a professional email with the following details:
        
        To: {recipient}
        Subject: {subject}
        Context/Request: {context}
        
        Relevant information from previous emails:
        {context_info}
        
        Write a well-structured, professional email:
        """
        
        return self.llm(prompt)
    
    def get_email_insights(self) -> Dict[str, Any]:
        """Get insights about email patterns"""
        # This would analyze the vector store for patterns
        # Simplified implementation
        return {
            "total_emails": "Available in vector store",
            "top_senders": "Analysis available via search",
            "common_topics": "Use search to find patterns",
            "priority_distribution": "High/Medium/Low classification available"
        }


def main():
    """Example usage of the Email Agent"""
    # Load configuration
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "gmail_credentials": "credentials.json"
    }
    
    if not config["openai_api_key"] or config["openai_api_key"] == "your-openai-api-key":
        print("Please set your OpenAI API key in the environment variable OPENAI_API_KEY")
        return
    
    try:
        # Initialize agent
        agent = EmailAgent(
            openai_api_key=config["openai_api_key"],
            gmail_credentials=config["gmail_credentials"]
        )
        
        # Sync recent emails
        print("Syncing recent emails...")
        agent.sync_emails(max_results=50)
        
        # Example queries
        queries = [
            "What are the most important emails from this week?",
            "Help me draft a follow-up email about the project deadline",
            "Summarize all emails about the quarterly review",
            "What emails need urgent attention?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            response = agent.process_query(query)
            print(f"Response: {response}")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
