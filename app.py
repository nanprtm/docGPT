from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update

# Import the necessary functions from utils.py
from utils import process_pdf, send_to_qdrant, qdrant_client, qa_ret, OpenAIEmbeddings

app = FastAPI()

async def start_command(update: Update, context):
    await update.message.reply_text('Hello! I am your PDF assistant bot. Send me a PDF file and then ask questions about it.')

async def help_command(update: Update, context):
    await update.message.reply_text('Commands:\n/start - Start the bot\n/help - Show this help message')

# Telegram bot token from environment variable
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
bot_app.add_handler(CommandHandler("start", start_command))
bot_app.add_handler(CommandHandler("help", help_command))

async def set_telegram_webhook():
    """Sets the Telegram webhook."""
    webhook_url = f"{os.getenv('YOUR_APP_URL')}/telegram/webhook"
    await bot_app.bot.set_webhook(webhook_url)

# Run the startup function to set the webhook
app.add_event_handler("startup", set_telegram_webhook)

# Handle Telegram updates via webhook
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    update = Update.de_json(await request.json(), bot_app.bot)
    await bot_app.process_update(update)
    return {"status": "ok"}

# Store for the Telegram bot application
#bot_app = None

# Frontend URL
FRONTEND_URL = os.getenv("FRONTEND_URL") 

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", FRONTEND_URL],  # Allow requests from your React app (adjust domain if necessary)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define a model for the question API

from pydantic import BaseModel
class QuestionRequest(BaseModel):
    question: str

# Endpoint to upload a PDF and process it, sending to Qdrant
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file, process it, and store in the vector DB.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Process the PDF to get document chunks and embeddings
        document_chunks = process_pdf(temp_file_path)

        # Create the embedding model (e.g., OpenAIEmbeddings)
        embedding_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),  # Assuming you're using env vars
            model="text-embedding-ada-002"
        )

        # Send the document chunks (with embeddings) to Qdrant
        success = send_to_qdrant(document_chunks, embedding_model)

        # Remove the temporary file after processing
        os.remove(temp_file_path)

        if success:
            return {"message": "PDF successfully processed and stored in vector DB"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store PDF in vector DB")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# Endpoint to ask a question and retrieve the answer from the vector DB
@app.post("/ask-question/")
async def ask_question(question_request: QuestionRequest):
    """
    Endpoint to ask a question and retrieve a response from the stored document content.
    """
    try:
        # Retrieve the Qdrant vector store (assuming qdrant_client() gives you access to it)
        qdrant_store = qdrant_client()

        # Get the question from the request body
        question = question_request.question

        # Use the question-answer retrieval function to get the response
        response = qa_ret(qdrant_store, question)

        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")

#A simple health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Hello There!"}

# Telegram command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /start command"""
    welcome_message = (
        "👋 Welcome to your PDF Q&A Bot!\n\n"
        "Here's how to use me:\n"
        "1. Send me a PDF document 📄\n"
        "2. Once I process it, you can ask me any questions about its content ❓\n"
        "3. I'll do my best to answer based on the document content 🤓\n\n"
        "Use /help to see all available commands."
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /help command"""
    help_text = (
        "📚 Available Commands:\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "You can also:\n"
        "• Send a PDF file to process it\n"
        "• Ask questions about the processed PDF\n"
    )
    await update.message.reply_text(help_text)

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for PDF document messages"""
    try:
        # Send an acknowledgment message
        processing_message = await update.message.reply_text("Processing your PDF... 📄")
        
        # Get the file from Telegram
        pdf_file = await context.bot.get_file(update.message.document.file_id)
        
        # Download the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            await pdf_file.download_to_drive(temp_file.name)
            temp_file_path = temp_file.name

        # Process the PDF
        document_chunks = process_pdf(temp_file_path)
        
        # Create embedding model
        embedding_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        
        # Send to Qdrant
        success = send_to_qdrant(document_chunks, embedding_model)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        if success:
            await processing_message.edit_text(
                "✅ PDF processed successfully! You can now ask me questions about it."
            )
        else:
            await processing_message.edit_text(
                "❌ Sorry, I couldn't process your PDF. Please try again."
            )
            
    except Exception as e:
        await update.message.reply_text(
            f"❌ Sorry, an error occurred while processing your PDF: {str(e)}"
        )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for text messages (questions)"""
    try:
        # Send a "thinking" message
        thinking_message = await update.message.reply_text("🤔 Let me think about that...")
        
        # Get the question from the message
        question = update.message.text
        
        # Get the Qdrant vector store
        qdrant_store = qdrant_client()
        
        # Get the answer
        answer = qa_ret(qdrant_store, question)
        
        # Edit the thinking message with the answer
        await thinking_message.edit_text(f"🔍 {answer}")
        
    except Exception as e:
        await update.message.reply_text(
            "❌ Sorry, I couldn't process your question. Make sure you've sent a PDF first!"
        )
