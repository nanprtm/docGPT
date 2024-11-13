from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from telegram import Update
from telegram.ext import ContextTypes

# Import the necessary functions from utils.py
from utils import process_pdf, send_to_qdrant, qdrant_client, qa_ret, OpenAIEmbeddings

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