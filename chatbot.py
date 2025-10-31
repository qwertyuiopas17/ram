from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import os
import logging
import traceback
from datetime import datetime, timedelta, timezone
import json
import time
import threading
from typing import Dict, Any, Optional
import numpy as np
import atexit
# (In chatbot.py, after the import statements)
import re
from functools import wraps
import uuid
import base64
from io import BytesIO
import groq
from agora_token_builder import RtcTokenBuilder, RtmTokenBuilder
from werkzeug.middleware.proxy_fix import ProxyFix
sos_events ={}
# For simplicity, a global dictionary is used here.


# In chatbot.py, replace the clean_ai_response function

def clean_ai_response(text: str) -> str:
    """Enhanced response cleaning that preserves JSON structure while removing artifacts"""
    if not isinstance(text, str):
        return text
    
    # Remove common AI artifacts
    cleaned_text = text.replace('\\n', '\n').replace("SAHARA:", "").strip()
    
    # Try to detect if this is JSON - if so, don't filter lines
    try:
        json.loads(cleaned_text)
        return cleaned_text  # It's valid JSON, return as-is
    except json.JSONDecodeError:
        pass
    
    # For non-JSON text, filter instructional phrases
    instructional_phrases = [
        "your task is to", "your response must be only", "return only json",
        "you are an ai", "as an ai assistant", "i cannot", "i apologize"
    ]
    lines = cleaned_text.splitlines()
    filtered = [ln for ln in lines if not any(p in ln.lower() for p in instructional_phrases)]
    result = "\n".join(filtered).strip()
    
    # Remove leading/trailing quotes if present
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    
    return result
# Import enhanced database models
from enhanced_database_models import (
    db, User, Doctor, Appointment, HealthRecord, Pharmacy, MedicineOrder,
    ConversationTurn, UserSession, GrievanceReport, SystemMetrics, init_database, get_user_statistics,
    PushSubscription  # <-- Import the new table
)

# Import enhanced AI components with Ollama integration
from nlu_processor import ProgressiveNLUProcessor
from ko import ProgressiveResponseGenerator
# Remove crisis detector import and usage
# from optimized_crisis_detector import OptimizedCrisisDetector  # removed

from api_ollama_integration import sehat_sahara_client, groq_scout
from conversation_memory import conversation_memory # <--- ADD THIS LINE
# --- NEW IMPORTS FOR PUSH NOTIFICATIONS ---
from pywebpush import webpush, WebPushException
from py_vapid import Vapid
from apscheduler.schedulers.background import BackgroundScheduler

# --- Import the official Groq library ---
try:
    from groq import Groq, GroqError # Import the main client and potential error types
except ImportError:
    logging.error("❌ 'groq' library not installed. Please run 'pip install groq'. AI features disabled.")
    Groq = None # Define as None so checks below work
    GroqError = Exception # Use base Exception for error handling if import fails


# Configure comprehensive logging with multiple handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handlers
file_handler = logging.FileHandler('chatbot.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

error_handler = logging.FileHandler('system_errors.log', mode='a', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Add console handler with proper encoding
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Console handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(error_handler)
logger.addHandler(stream_handler)

# Get module logger
logger = logging.getLogger(__name__)

# 1. NEW Client for Groq Whisper Transcription (using GROQ_WHISPER_KEY)
groq_whisper_client = None
groq_whisper_key = os.getenv("GROQ_WHISPER_KEY")

if Groq: # Check if import succeeded
    try:
        if groq_whisper_key:
            # --- Directly instantiate the Groq client ---
            groq_whisper_client = Groq(api_key=groq_whisper_key)
            # You might add a simple test call here if needed to verify the key
            logger.info("✅ Groq Whisper client initialized successfully (using GROQ_WHISPER_KEY).")
        else:
            logger.warning("⚠️ GROQ_WHISPER_KEY not found. Groq Whisper transcription will be disabled.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Groq Whisper client: {e}", exc_info=True)
        groq_whisper_client = None
else:
     logger.error("❌ Groq library not found. Cannot initialize Groq Whisper client.")


# 2. NEW Client for Groq LLM Structuring (using GROQ_LLM_KEY)
groq_llm_client = None
groq_llm_key = os.getenv("GROQ_LLM_KEY")

if Groq: # Check if import succeeded
    try:
        if groq_llm_key:
            # --- Directly instantiate the Groq client ---
            groq_llm_client = Groq(api_key=groq_llm_key)
            # You might add a simple test call here if needed
            logger.info("✅ Groq LLM client initialized successfully (using GROQ_LLM_KEY).")
        else:
            logger.warning("⚠️ GROQ_LLM_KEY not found. AI prescription structuring (Groq) will be disabled.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Groq LLM client: {e}", exc_info=True)
        groq_llm_client = None
else:
     logger.error("❌ Groq library not found. Cannot initialize Groq LLM client.")

# Initialize Flask application with enhanced configuration
app = Flask(__name__)
# Enhanced CORS configuration for better compatibility
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app, supports_credentials=True, resources={
    r"/*": {  # Covers ALL routes including /v1/*
        "origins": [
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "http://localhost:3000",
            "https://saharasaathi.netlify.app",
            "https://sahara-sathi.onrender.com",
            "https://sehat-sahara.onrender.com",
            "https://visionary-heliotrope-8203e0.netlify.app"  # Allow all origins for static files
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# Load Agora credentials securely from environment variables
AGORA_APP_ID = os.environ.get("AGORA_APP_ID")
AGORA_APP_CERTIFICATE = os.environ.get("AGORA_APP_CERTIFICATE")
# Set token expiration time in seconds (e.g., 1 hour)
TOKEN_EXPIRATION_SEC = 3600
# Enhanced security configuration
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config.update(
    SESSION_COOKIE_SECURE=True,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='None',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24)
)

# Enhanced database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
models_path = os.path.join(basedir, 'models')
logs_path = os.path.join(basedir, 'logs')

# Ensure all directories exist
for path in [instance_path, models_path, logs_path]:
    os.makedirs(path, exist_ok=True)

# --- THIS IS THE CORRECTED CODE BLOCK ---

# First, determine the correct database URI
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith('postgres://'):
    # This path is for Render (PostgreSQL)
    db_uri = database_url.replace('postgres://', 'postgresql://', 1)
else:
    # This path is for your local computer (SQLite)
    db_uri = f'sqlite:///{os.path.join(instance_path, "enhanced_chatbot.db")}'

# Now, update the app configuration
app.config.update({
    'SQLALCHEMY_DATABASE_URI': db_uri,
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'SQLALCHEMY_ENGINE_OPTIONS': {
        'pool_timeout': 30,
        'pool_recycle': 300,
        'pool_pre_ping': True,
        'echo': False
    }
})

# Initialize database
db.init_app(app)
# --- VAPID KEYS SETUP FOR PUSH NOTIFICATIONS ---
# These keys identify your server to push services.
# In a real production environment, generate these once and store them securely
# as environment variables.
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
VAPID_CLAIM_EMAIL = "dedpull2005@outlook.com"

if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
    logger.critical("FATAL ERROR: VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY are not set in the environment.")
    logger.critical("Please generate keys using a VAPID key generator and add them to your Render Environment Variables.")
    # Exit the application if keys are not found, as push notifications cannot work.
    exit("VAPID keys are not configured. Application cannot start.")

# --- BACKGROUND SCHEDULER FOR SENDING NOTIFICATIONS ---
scheduler = BackgroundScheduler(daemon=True)

# Replace your entire function with this one
# In chatbot.py

# Replace your entire function with this one
def check_and_send_reminders():
    """The background job that checks for due reminders, sends push notifications, and reschedules."""
    # --- ADDED: Ensure app context for database operations ---
    with app.app_context():
        # --- ADDED: Load memory at the start of the job ---
        memory_loaded = conversation_memory.load_from_file(os.path.join(models_path, 'conversation_memory.json'))
        if not memory_loaded:
             logger.error("Failed to load conversation memory in reminder job. Aborting check.")
             return # Cannot proceed without memory

        try:
            now_utc = datetime.now(timezone.utc)
            reminders_updated = False # Flag to check if we need to save

            # Iterate through a copy of profiles in case memory changes during iteration (less likely here but safer)
            for user_id, profile in list(conversation_memory.user_profiles.items()):
                # Iterate through a copy of reminders for safe modification
                for reminder in list(profile.medicine_reminders):
                    # Check if reminder is enabled and has a next alert time
                    if reminder.get('reminder_enabled', True) and reminder.get('next_alert_utc') and not reminder.get('alert_sent'):
                        try:
                            alert_time_utc = datetime.fromisoformat(reminder['next_alert_utc'])
                        except ValueError:
                             logger.error(f"Invalid ISO format for next_alert_utc for user {user_id}, reminder {reminder.get('medicine_name')}. Skipping.")
                             continue # Skip this reminder

                        # Check if the alert time is now or in the past
                        if alert_time_utc <= now_utc:
                            logger.info(f"Alert due for user {profile.user_id} for medicine {reminder['medicine_name']} at {alert_time_utc}")

                            user_db = User.query.filter_by(patient_id=profile.user_id).first()
                            if not user_db:
                                logger.warning(f"User DB record not found for patient_id {profile.user_id}. Cannot send reminder.")
                                continue # Skip if user doesn't exist in DB

                            subscriptions = PushSubscription.query.filter_by(user_id=user_db.id).all()
                            if not subscriptions:
                                logger.warning(f"No push subscriptions found for user {profile.user_id}. Marking alert as sent.")
                                # --- START: Reschedule even if no subscription ---
                                user_timezone = reminder.get("timezone", "UTC")
                                times_list = reminder.get("times", [])
                                if times_list:
                                    reminder['next_alert_utc'] = conversation_memory._calculate_next_utc_timestamp(times_list, user_timezone)
                                    reminder['alert_sent'] = False # Reset flag for the *next* alert
                                    logger.info(f"Rescheduled reminder (no subs) for {reminder['medicine_name']} to {reminder['next_alert_utc']}")
                                    reminders_updated = True
                                else:
                                    reminder['alert_sent'] = True # Cannot reschedule without times
                                    reminders_updated = True
                                # --- END: Reschedule even if no subscription ---
                                continue

                            push_payload = json.dumps({
                                "title": "💊 Sehat Sahara Reminder",
                                "options": {
                                    "body": f"It's time to take: {reminder['medicine_name']} ({reminder.get('dosage', 'N/A')})",
                                    # --- ADDED: More specific icon ---
                                    "icon": "https://i.ibb.co/bmdxHqN/pills.png",
                                    "actions": [
                                        {"action": "mark-taken", "title": "Mark as Taken"},
                                        {"action": "close", "title": "Close"}
                                    ],
                                    "data": {
                                        # --- ADDED: Send necessary data to frontend ---
                                        "action": "mark-taken",
                                        "medicineName": reminder['medicine_name'],
                                        "userId": profile.user_id
                                    },
                                    # --- ADDED: Tag to allow replacing pending notifications ---
                                    "tag": f"med-reminder-{profile.user_id}-{reminder['medicine_name'].replace(' ','-')}"
                                }
                            })

                            sent_successfully = False
                            for sub in subscriptions:
                                try:
                                    webpush(
                                        subscription_info=json.loads(sub.subscription_info),
                                        data=push_payload,
                                        vapid_private_key=VAPID_PRIVATE_KEY,
                                        vapid_claims={"sub": "mailto:" + VAPID_CLAIM_EMAIL}
                                    )
                                    logger.info(f"Push notification sent successfully to a device for user {profile.user_id}")
                                    sent_successfully = True
                                except WebPushException as ex:
                                    logger.error(f"Failed to send push notification: {ex}")
                                    # Handle expired/invalid subscriptions
                                    if ex.response and ex.response.status_code in [404, 410]:
                                        logger.info(f"Deleting expired subscription for user {profile.user_id}")
                                        db.session.delete(sub)
                                        # Commit immediately to remove invalid sub
                                        db.session.commit()
                                except Exception as push_err:
                                     logger.error(f"Unexpected error sending push: {push_err}")


                            # --- START: Modified Rescheduling Logic ---
                            if sent_successfully:
                                logger.info(f"Successfully sent alert for {reminder['medicine_name']} to user {profile.user_id}. Now rescheduling.")
                            else:
                                logger.warning(f"Failed to send alert for {reminder['medicine_name']} to user {profile.user_id} (no valid subscriptions?). Still rescheduling.")

                            # Calculate the next alert time REGARDLESS of send success
                            user_timezone = reminder.get("timezone", "UTC")
                            times_list = reminder.get("times", [])

                            # Optional: Add duration check here later if needed
                            start_date_str = reminder.get('start_date')
                            duration_days = reminder.get('duration_days')
                            if start_date_str and duration_days:
                                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                                end_date = start_date + timedelta(days=duration_days)
                                if datetime.now().date() >= end_date:
                                    logger.info(f"Reminder duration ended for {reminder['medicine_name']}. Disabling.")
                                    reminder['reminder_enabled'] = False
                                    reminders_updated = True
                                    continue # Skip rescheduling

                            if times_list:
                                next_alert_iso = conversation_memory._calculate_next_utc_timestamp(times_list, user_timezone)
                                reminder['next_alert_utc'] = next_alert_iso
                                reminder['alert_sent'] = False # Reset flag for the *next* schedule
                                logger.info(f"Rescheduled reminder for {reminder['medicine_name']} to {next_alert_iso}")
                                reminders_updated = True
                            else:
                                # Cannot reschedule without a time list, mark as sent to prevent re-sending
                                reminder['alert_sent'] = True
                                logger.warning(f"Reminder for {reminder['medicine_name']} has no times_list, cannot reschedule. Marked as sent.")
                                reminders_updated = True
                            # --- END: Modified Rescheduling Logic ---

            # --- Save memory ONCE after checking all users/reminders ---
            if reminders_updated:
                logger.info("Saving updated reminder schedules to memory file.")
                conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))

        except Exception as e:
            logger.error(f"Error in background reminder job: {e}", exc_info=True)
            # Rollback potential DB changes if error occurred during subscription deletion
            db.session.rollback()


# Start the background job
scheduler.add_job(check_and_send_reminders, 'interval', minutes=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# Global system state tracking
system_state = {
    'startup_time': datetime.now(),
    'total_requests': 0,
    'successful_responses': 0,
    'error_count': 0,
    'appointments_booked': 0,
    'sos_triggered': 0,
    'llama_responses': 0,
    'fallback_responses': 0
}



# Thread lock for system state updates
state_lock = threading.Lock()

# Initialize enhanced AI components with comprehensive error handling
def initialize_ai_components():
    """Initialize all AI components with Ollama integration and proper error handling"""
    global nlu_processor, response_generator, conversation_memory
    global system_status

    logger.info("🚀 Initializing Sehat Sahara Health Assistant...")

    # Model file paths
    nlu_model_path = os.path.join(models_path, 'progressive_nlu_model.pkl')
    memory_model_path = os.path.join(models_path, 'progressive_memory.pkl')
    conversation_memory_path = os.path.join(models_path, 'conversation_memory.json')

    system_status = {
        'nlu_processor': False,
        'response_generator': False,
        'conversation_memory': False,
        'database': False,
        'ollama_llama3': sehat_sahara_client.is_available  # reuse flag to show API availability
    }

    try:
        # Check Ollama Llama 3 availability (using Sehat Sahara client's flag)
        logger.info("🦙 Checking Sehat Sahara API availability...")
        system_status['ollama_llama3'] = sehat_sahara_client.is_available

        if sehat_sahara_client.is_available:
            logger.info("✅ Sehat Sahara API is available and ready for AI-enhanced responses")
        else:
            logger.info("⚠️ Sehat Sahara API not available - using rule-based responses with fallback")


        # Initialize NLU Processor with OpenRouter API support
        logger.info("🧠 Initializing Progressive NLU Processor with OpenRouter AI...")
        nlu_processor = ProgressiveNLUProcessor(
            model_path=nlu_model_path
        )
        system_status['nlu_processor'] = True
        logger.info("✅ NLU Processor initialized successfully")

        

        # Initialize Response Generator
        logger.info("💬 Initializing Progressive Response Generator...")
        response_generator = ProgressiveResponseGenerator()
        system_status['response_generator'] = True
        logger.info("✅ Response Generator initialized successfully")

        # Initialize Conversation Memory
        logger.info("Initializing Progressive Conversation Memory...")
        from conversation_memory import ProgressiveConversationMemory
        # Create a module-level instance instead of using global
        if not hasattr(initialize_ai_components, '_conversation_memory'):
            initialize_ai_components._conversation_memory = ProgressiveConversationMemory()
        system_status['conversation_memory'] = True
        logger.info("Conversation Memory initialized successfully")

        # Load conversation memory
        logger.info("💾 Loading conversation memory...")
        try:
            if os.path.exists(conversation_memory_path):
                if initialize_ai_components._conversation_memory.load_from_file(conversation_memory_path):
                    logger.info("✅ Conversation memory loaded successfully")
                else:
                    logger.warning("⚠️ Failed to load conversation memory, starting fresh")
            else:
                logger.info("💾 No existing conversation memory file found, starting fresh")
        except Exception as e:
            logger.error(f"❌ Error loading conversation memory: {e}")

        logger.info("✅ All AI components initialized for Sehat Sahara.")
        return True

    except Exception as e:
        logger.error(f"❌ Critical error initializing AI components: {e}")
        logger.error(traceback.format_exc())

        # Initialize minimal fallback components
        try:
            logger.info("🔄 Attempting to initialize fallback components...")
            openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
            nlu_processor = ProgressiveNLUProcessor(openrouter_api_key=openrouter_api_key)
            response_generator = ProgressiveResponseGenerator()
            from conversation_memory import ProgressiveConversationMemory
            if not hasattr(initialize_ai_components, '_conversation_memory'):
                initialize_ai_components._conversation_memory = ProgressiveConversationMemory()
            logger.info("⚠️ Fallback components initialized (limited functionality)")
            return False
        except Exception as fallback_error:
            logger.error(f"❌ Failed to initialize even fallback components: {fallback_error}")
            nlu_processor = None
            response_generator = None
            return False

# Initialize system
ai_initialized = initialize_ai_components()

# Database initialization with app context
with app.app_context():
    try:
        init_database(app)
        system_status['database'] = True
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        system_status['database'] = False

# Utility functions for system management
def update_system_state(operation: str, success: bool = True, **kwargs):
    """Thread-safe system state updates"""
    with state_lock:
        system_state['total_requests'] += 1
        if success:
            system_state['successful_responses'] += 1
        else:
            system_state['error_count'] += 1

        # Update specific counters
        for key, value in kwargs.items():
            if key in system_state:
                system_state[key] += value

def save_all_models():
    """Save all AI models with comprehensive error handling"""
    try:
        if nlu_processor and system_status['nlu_processor']:
            nlu_processor.save_nlu_model(os.path.join(models_path, 'progressive_nlu_model.pkl'))
            logger.info("NLU model saved")

        if hasattr(initialize_ai_components, '_conversation_memory') and system_status['conversation_memory']:
            initialize_ai_components._conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))
            logger.info("Conversation memory saved")

        # Crisis detector model saving removed as it's no longer used

        logger.info("All models saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return False

def track_system_metrics():
    """Tracks and updates system-wide metrics for the Admin dashboard."""
    try:
        today = datetime.now().date()

        # Avoid creating duplicate metrics for the same day
        if SystemMetrics.query.filter_by(metrics_date=today).first():
            logger.info(f"Metrics for {today} already exist. Skipping.")
            return

        start_of_day = datetime.combine(today, datetime.min.time())

        # Calculate metrics for today
        active_users = User.query.filter(User.last_login >= start_of_day).count()
        new_users = User.query.filter(User.created_at >= start_of_day).count()
        total_convos = ConversationTurn.query.filter(ConversationTurn.timestamp >= start_of_day).count()
        appts_booked = Appointment.query.filter(Appointment.created_at >= start_of_day).count()
        orders_placed = MedicineOrder.query.filter(MedicineOrder.created_at >= start_of_day).count()

        # You would add grievances and prescriptions issued counts here as well

        metrics = SystemMetrics(
            metrics_date=today,
            total_active_users=active_users,
            new_users_registered=new_users,
            total_conversations=total_convos,
            appointments_booked=appts_booked,
            orders_placed=orders_placed
        )

        db.session.add(metrics)
        db.session.commit()
        logger.info(f"✅ Admin System Metrics updated for {today}")

    except Exception as e:
        logger.error(f"❌ Error tracking system metrics: {e}")
        db.session.rollback()

def get_current_user():
    """Security helper to get current authenticated user for any role."""
    try:
        user_pk_id = session.get('user_id')
        role = session.get('role')

        if not user_pk_id or not role:
            return None

        if role == 'doctor':
            doctor_id_str = session.get('doctor_id')
            return Doctor.query.filter_by(doctor_id=doctor_id_str, is_active=True).first()

        elif role == 'pharmacy':
            pharmacy_id_str = session.get('pharmacy_id')
            return Pharmacy.query.filter_by(pharmacy_id=pharmacy_id_str, is_active=True).first()
        
        elif role in ['patient', 'saathi', 'admin']:
            user = User.query.filter_by(id=user_pk_id, is_active=True).first()
            if user and user.role == role:
                return user
        
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error in get_current_user: {e}")
        return None

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        # Assuming User model has a 'role' attribute, default to 'patient' if not present
        if getattr(user, "role", "patient") != "admin":
            return jsonify({"error": "Forbidden"}), 403
        return f(*args, **kwargs)
    return wrapper

def create_user_session(user: User, request_info: dict):
    """Create and track user session"""
    try:
        user_session = UserSession(
            user_id=user.id,
            ip_address=request_info.get('remote_addr', '')[:45],
            user_agent=request_info.get('user_agent', '')[:500],
            device_type=determine_device_type(request_info.get('user_agent', ''))
        )

        db.session.add(user_session)
        db.session.commit()

        # Store session ID for later reference
        session['session_record_id'] = user_session.id

    except Exception as e:
        logger.error(f"Error creating user session: {e}")

def determine_device_type(user_agent: str) -> str:
    """Determine device type from user agent"""
    user_agent = user_agent.lower()
    if any(mobile in user_agent for mobile in ['mobile', 'android', 'iphone']):
        return 'mobile'
    elif 'tablet' in user_agent or 'ipad' in user_agent:
        return 'tablet'
    else:
        return 'desktop'

def end_user_session():
    """End current user session"""
    try:
        session_id = session.get('session_record_id')
        if session_id:
            user_session = UserSession.query.get(session_id)
            if user_session:
                user_session.end_session()
                db.session.commit()
    except Exception as e:
        logger.error(f"Error ending user session: {e}")

def convert_numpy_types(obj):
    """Recursively converts numpy types to native Python types in a dictionary."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Route to serve uploaded files
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    """Serve uploaded files"""
    uploads_dir = os.path.join(basedir, 'uploads')
    return send_from_directory(uploads_dir, filename)
# Enhanced API Routes with comprehensive functionality
@app.route("/v1/register", methods=["POST"])
def register():
    try:
        update_system_state('register')
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data provided."}), 400

        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        full_name = data.get("fullName", "").strip()
        phone_number = data.get("phoneNumber", "").strip()
        village = data.get("village", "").strip()
        district = data.get("district", "").strip()
        state = data.get("state", "Punjab").strip()
        pincode = data.get("pincode", "").strip()
        preferred_language = data.get("preferredLanguage", "hi").strip()

        # Validate required fields
        errors = []
        if not full_name: errors.append("Full name is required")
        if not email or "@" not in email: errors.append("Valid email is required")
        if not password or len(password) < 8: errors.append("Password must be at least 8 characters")
        if errors:
            return jsonify({"success": False, "message": "Fix the errors below.", "errors": errors}), 400

        # Check if the full name or email is already taken
        existing_user = User.query.filter((User.full_name == full_name) | (User.email == email)).first()
        if existing_user:
            return jsonify({"success": False, "message": "A user with this name or email already exists."}), 409

        # Auto-generate patient ID
        last_user = User.query.order_by(User.id.desc()).first()
        seq = last_user.id + 1 if last_user else 1
        patient_id = f"PAT{str(seq).zfill(6)}"

        new_user = User(
            patient_id=patient_id,
            email=email,
            full_name=full_name,
            phone_number=phone_number,
            village=village,
            district=district,
            state=state,
            pincode=pincode,
            preferred_language=preferred_language
        )
        new_user.set_password(password)
        # --- FIX: Explicitly set the role for new patient registrations to prevent DB errors ---
        new_user.role = 'patient'

        db.session.add(new_user)
        db.session.commit()

        logger.info(f"✅ New patient registered: {full_name} ({patient_id})")
        return jsonify({
            "success": True,
            "message": f"Welcome {full_name}! Your account has been created.",
            "patientId": patient_id,
            "fullName": full_name
        }), 201

    except Exception as e:
        # --- FIX: Added more detailed logging for registration errors ---
        logger.error(f"❌ Registration error for data: {data}. Exception: {e}")
        logger.error(traceback.format_exc())
        db.session.rollback()
        update_system_state('register', success=False)
        return jsonify({"success": False, "message": "An internal error occurred during registration. Please try again."}), 500

# In chatbot.py, replace your /v1/login function with this corrected version

@app.route("/v1/login", methods=["POST"])
def login():
    try:
        data = request.get_json() or {}
        login_identifier = data.get("patientId", "").strip()
        password = data.get("password", "")
        role = data.get("role", "patient").strip().lower() # The role selected in the dropdown

        if not login_identifier or not password:
            return jsonify({"success": False, "message": "Identifier and password are required."}), 400

        user_obj = None
        if role == 'doctor':
            user_obj = Doctor.query.filter(
                (Doctor.email == login_identifier.lower()) | (Doctor.doctor_id == login_identifier.upper())
            ).first()
        elif role == 'pharmacy':
            user_obj = Pharmacy.query.filter(
                (Pharmacy.email == login_identifier.lower()) | (Pharmacy.pharmacy_id == login_identifier.upper())
            ).first()
        else: # Handles patient, saathi, and admin from the User table
            user_obj = User.query.filter(
                (User.email == login_identifier.lower()) | (User.patient_id == login_identifier.upper())
            ).first()

        # --- THIS IS THE NEW SECURITY CHECK ---
        # After finding a user, we MUST check if their actual role matches the role they claimed to be.
        # For Doctors and Pharmacies, the query itself already filters by the correct table.
        # For the User table, we need this explicit check.
        if user_obj and role in ['patient', 'saathi', 'admin'] and user_obj.role != role:
            logger.warning(f"Role mismatch attempt for user: '{login_identifier}'. Tried to log in as '{role}' but is actually a '{user_obj.role}'.")
            # Return a generic error to avoid revealing which part (user/pass/role) was wrong.
            return jsonify({"success": False, "message": "Invalid credentials or role selection."}), 401
        # --- END OF SECURITY CHECK ---

        # If the roles match (or for doctor/pharmacy), now we check the password.
        if user_obj and user_obj.is_active and user_obj.check_password(password):
            # The rest of your successful login logic goes here, no changes needed.
            session.permanent = True
            user_type = role # Use the verified role from the request
            session['user_id'] = user_obj.id
            session['role'] = user_type
            session['login_time'] = datetime.now().isoformat()
            
            response_data = {"success": True}
            user_data = {}

            if user_type == 'doctor':
                session['doctor_id'] = user_obj.doctor_id
                response_data['message'] = f"Welcome back, Dr. {user_obj.full_name}!"
                user_data = {"patientId": user_obj.doctor_id, "username": user_obj.full_name, "role": "doctor"}
            elif user_type == 'pharmacy':
                session['pharmacy_id'] = user_obj.pharmacy_id
                response_data['message'] = f"Welcome back, {user_obj.name}!"
                response_data['redirect'] = "/store.html"
                user_data = {"pharmacyId": user_obj.pharmacy_id, "username": user_obj.name, "role": "pharmacy"}
            else: # patient, saathi, admin
                user_obj.update_last_login()
                db.session.commit()
                response_data['message'] = f"Welcome back, {user_obj.full_name}!"
                user_data = {
                    "patientId": user_obj.patient_id, 
                    "fullName": user_obj.full_name,
                    "role": user_type
                }

            response_data['user'] = user_data
            logger.info(f"✅ {user_type.capitalize()} login successful: {login_identifier}")
            return jsonify(response_data)

        # This will now catch password mismatches AND role mismatches.
        logger.warning(f"⚠️ Failed login attempt for identifier: '{login_identifier}' as {role}")
        return jsonify({"success": False, "message": "Invalid credentials or role selection."}), 401

    except Exception as e:
        logger.error(f"❌ Login error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Login failed due to a server error."}), 500


@app.route("/v1/logout", methods=["POST"])
def logout():
    """Enhanced user logout with session cleanup"""
    try:
        # End user session tracking
        end_user_session()

        # Clear session
        session.clear()

        logger.info("✅ User logged out successfully")

        return jsonify({
            "success": True,
            "message": "You have been logged out successfully."
        })

    except Exception as e:
        logger.error(f"❌ Logout error: {e}")
        return jsonify({
            "success": True,  # Always succeed logout for security
            "message": "Logged out successfully."
        })
@app.route("/v1/button-action", methods=["POST"])
def handle_button_action():
    data = request.get_json() or {}
    user_id = data.get("userId")
    button_action = data.get("buttonAction")

    logger.info(f"Button action triggered for user {user_id}: {button_action}")

    # This endpoint is now just for logging/analytics.
    # The old, crashing logic has been removed.

    return jsonify({
        "success": True,
        "message": f"Action {button_action} acknowledged."
    })

@app.route("/v1/user-progress", methods=["POST"])
def get_user_progress():
    """Get user's progress and interactive state"""
    try:
        data = request.get_json() or {}
        user_id = data.get("userId", "").strip()

        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        # Get progress summary from conversation memory
        if hasattr(initialize_ai_components, '_conversation_memory'):
            progress_summary = initialize_ai_components._conversation_memory.get_user_progress_summary(user_id)
            followup_needed = initialize_ai_components._conversation_memory.check_post_appointment_followup(user_id)
        else:
            progress_summary = {}
            followup_needed = {'needed': False}

        return jsonify({
            "success": True,
            "progress": progress_summary,
            "followup": followup_needed,
            "interactive_buttons": progress_summary.get('interactive_buttons', {})
        })

    except Exception as e:
        logger.error(f"Get user progress error: {e}")
        return jsonify({"error": "Failed to get user progress"}), 500

@app.route("/v1/post-appointment-feedback", methods=["POST"])
def handle_post_appointment_feedback():
    """
    New endpoint to handle post-appointment feedback with AI-generated contextual responses
    """
    try:
        data = request.get_json() or {}
        user_id = data.get("userId", "").strip()
        feedback = data.get("feedback", "").strip().lower()
        language = data.get("language", "en")

        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        # Get conversation history for context
        if hasattr(initialize_ai_components, '_conversation_memory'):
            history_turns = initialize_ai_components._conversation_memory.get_conversation_context(user_id, turns=4)
            nlu_history = []
            for turn in history_turns:
                nlu_history.append({'role': 'user', 'content': turn.get('user_message', '')})
                try:
                    bot_response_json = json.loads(turn.get('bot_response', '{}'))
                    bot_content = bot_response_json.get('response', '')
                except (json.JSONDecodeError, AttributeError):
                    bot_content = turn.get('bot_response', '')
                nlu_history.append({'role': 'assistant', 'content': bot_content})
        else:
            nlu_history = []

        # Use AI to generate contextual response
        response_text = None
        action = "CONTINUE_FOLLOWUP"

        if sehat_sahara_client and sehat_sahara_client.is_available:
            ai_response_str = sehat_sahara_client.generate_sehatsahara_response(
                user_message=feedback,
                user_intent='post_appointment_followup',
                conversation_stage='followup',
                severity_score=0.3,
                context_history=nlu_history,
                language=language
            )

            if ai_response_str:
                try:
                    ai_response = json.loads(ai_response_str)
                    response_text = ai_response.get('response')
                    action = ai_response.get('action', 'CONTINUE_FOLLOWUP')
                except json.JSONDecodeError:
                    pass

        # Fallback responses if AI fails
        if not response_text:
            is_positive = any(word in feedback for word in ['better', 'good', 'well', 'improved', 'accha', 'changa', 'ਸੁਧਾਰ'])

            if is_positive:
                positive_responses = {
                    'en': "I'm so glad you're feeling better! Please continue following your doctor's advice and take your medications as prescribed. Remember to complete the full course of treatment.",
                    'hi': "आप बेहतर महसूस कर रहे हैं यह सुनकर बहुत खुशी हुई! कृपया डॉक्टर की सलाह फॉलो करते रहें और दवाइयां बताई गई अनुसार लें। इलाज का पूरा कोर्स पूरा करें।",
                    'pa': "ਤੁਸੀਂ ਬਿਹਤਰ ਮਹਿਸੂਸ ਕਰ ਰਹੇ ਹੋ ਇਹ ਸੁਣਕੇ ਬਹੁਤ ਖੁਸ਼ੀ ਹੋਈ! ਕਿਰਪਾ ਕਰਕੇ ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਫੌਲੋ ਕਰਦੇ ਰਹੋ ਅਤੇ ਦਵਾਈਆਂ ਨਿਰਧਾਰਤ ਅਨੁਸਾਰ ਲੈਂਦੇ ਰਹੋ। ਇਲਾਜ ਦਾ ਪੂਰਾ ਕੋਰਸ ਪੂਰਾ ਕਰੋ।"
                }
                response_text = positive_responses.get(language, positive_responses['en'])
                action = "SHOW_PRESCRIPTION_REMINDER"

            else:
                negative_responses = {
                    'en': "I'm sorry to hear you're not feeling better. Please review your prescription carefully and make sure you're following the doctor's instructions. If symptoms persist, consider booking another appointment.",
                    'hi': "आप बेहतर महसूस नहीं कर रहे हैं यह सुनकर दुख हुआ। कृपया अपनी प्रिस्क्रिप्शन ध्यान से देखें और सुनिश्चित करें कि आप डॉक्टर की instructions फॉलो कर रहे हैं। अगर symptoms बने रहें, तो दूसरा appointment बुक करें।",
                    'pa': "ਤੁਸੀਂ ਬਿਹਤਰ ਮਹਿਸੂਸ ਨਹੀਂ ਕਰ ਰਹੇ ਹੋ ਇਹ ਸੁਣਕੇ ਦੁੱਖ ਹੋਇਆ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਪ੍ਰਿਸਕ੍ਰਿਪਸ਼ਨ ਧਿਆਨ ਨਾਲ ਵੇਖੋ ਅਤੇ ਯਕੀਨੀ ਬਣਾਓ ਕਿ ਤੁਸੀਂ ਡਾਕਟਰ ਦੀਆਂ ਹਦਾਇਤਾਂ ਫੌਲੋ ਕਰ ਰਹੇ ਹੋ। ਜੇਕਰ ਅਲਾਮਤਾਂ ਬਣੀਆਂ ਰਹਿਣ, ਤਾਂ ਦੂਜੀ ਅਪਾਇੰਟਮੈਂਟ ਬੁਕ ਕਰੋ।"
                }
                response_text = negative_responses.get(language, negative_responses['en'])
                action = "SHOW_PRESCRIPTION_SUMMARY"

        if hasattr(initialize_ai_components, '_conversation_memory'):
            initialize_ai_components._conversation_memory.complete_post_appointment_feedback(user_id)

        return jsonify({
            "success": True,
            "response": response_text,
            "action": action,
            "parameters": {},
            "feedback_recorded": True,
            "language": language
        })

    except Exception as e:
        logger.error(f"Post-appointment feedback error: {e}")
        return jsonify({"error": "Failed to process feedback"}), 500
    
@app.route("/v1/vapid-public-key", methods=["GET"])
def get_vapid_public_key():
    """Provide the frontend with the server's public key."""
    return jsonify({"publicKey": VAPID_PUBLIC_KEY})

@app.route("/v1/subscribe", methods=["POST"])
def subscribe():
    """Subscribe a user's device for push notifications."""
    try:
        data = request.get_json() or {}
        subscription_info = data.get("subscription")
        user_id_str = data.get("userId")

        if not subscription_info or not user_id_str:
            return jsonify({"error": "Subscription info and userId are required"}), 400

        user = User.query.filter_by(patient_id=user_id_str).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Check if this exact subscription already exists to avoid duplicates
        endpoint = subscription_info.get('endpoint')
        existing_sub = PushSubscription.query.filter(PushSubscription.subscription_info.like(f'%"{endpoint}"%')).first()

        if existing_sub:
             logger.info(f"Subscription for endpoint {endpoint} already exists for user {user.patient_id}.")
             return jsonify({"success": True, "message": "Already subscribed"})

        new_sub = PushSubscription(
            user_id=user.id,
            subscription_info=json.dumps(subscription_info)
        )
        db.session.add(new_sub)
        db.session.commit()

        logger.info(f"New push subscription saved for user {user.patient_id}")
        return jsonify({"success": True, "message": "Subscribed successfully"}), 201

    except Exception as e:
        logger.error(f"Subscription error: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Failed to subscribe"}), 500
    
@app.route("/v1/doctor/generate-prescription-from-speech", methods=["POST"])
def generate_prescription_from_speech():
    """
    Handles AI prescription generation: Groq Whisper transcribes, Groq LLM structures.
    Requires GROQ_WHISPER_KEY and GROQ_LLM_KEY environment variables.
    """
    # 1. Authentication and Input Validation
    try:
        doctor = get_current_user()
        if not doctor or not isinstance(doctor, Doctor):
            logger.warning("Unauthorized attempt: Doctor role required.")
            return jsonify({"success": False, "message": "Doctor authentication required"}), 401

        patient_id_str = request.form.get("patientId")
        if not patient_id_str:
            logger.warning("Missing 'patientId' in form data.")
            return jsonify({"success": False, "message": "Patient ID is required"}), 400

        user = User.query.filter_by(patient_id=patient_id_str, is_active=True).first()
        if not user:
            logger.warning(f"Patient not found: {patient_id_str}")
            return jsonify({"success": False, "message": "Patient not found"}), 404

        if 'audio' not in request.files:
            logger.warning("No 'audio' file part in request.")
            return jsonify({"success": False, "message": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.warning("Empty filename for audio file.")
            return jsonify({"success": False, "message": "No selected audio file"}), 400

    except Exception as e:
        logger.error(f"Error during initial validation: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Invalid request data."}), 400

    # 2. Transcription using Groq Whisper Client
    if not groq_whisper_client:
        logger.error("Groq Whisper client is not initialized.")
        return jsonify({"success": False, "message": "Speech transcription service is unavailable."}), 503

    transcription_text = ""
    try:
        audio_data = audio_file.read() # Read audio file bytes

        logger.info(f"Sending audio ({len(audio_data)} bytes) to Groq Whisper for patient {patient_id_str}...")
        transcript_response = groq_whisper_client.audio.transcriptions.create(
             model="whisper-large-v3",
             file=(audio_file.filename, audio_data), # Pass as tuple (filename, bytes)
             response_format="json" # Get JSON to extract 'text' easily
             # language="en" # Optional: Add language hint if needed
        )

        transcription_text = transcript_response.text
        if not transcription_text or transcription_text.strip() == "":
             logger.warning("Groq Whisper returned empty transcription.")
             return jsonify({"success": False, "message": "Could not understand speech or audio was empty (Groq Whisper)."}), 400

        logger.info(f"Groq Whisper transcription success: '{transcription_text}'")

    except GroqError as e:
        logger.error(f"Groq API error during transcription: {e.status_code} - {e}", exc_info=True)
        status_code = getattr(e, 'status_code', 500)
        message = f"Groq transcription error: {e}"
        if status_code == 401: message = "Groq Whisper API key invalid/missing (GROQ_WHISPER_KEY)."
        if status_code == 429: message = "Groq Whisper API rate limit reached."
        return jsonify({"success": False, "message": message}), status_code if status_code != 500 else 503
    except Exception as e:
        logger.error(f"Unexpected error during Groq Whisper transcription: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to transcribe speech."}), 500

    # 3. Structuring using Groq LLM Client
    if not groq_llm_client:
         logger.error("Groq LLM client is not initialized.")
         return jsonify({"success": False, "message": "AI structuring service (Groq) is unavailable."}), 503

    structured_prescription = {}
    raw_llm_output = "" # Initialize for error logging
    try:
        prompt = f"""Extract the primary medicine name, its dosage instructions (including frequency, duration, relation to food if mentioned), and any general advice from the following doctor's transcription. Respond ONLY with a valid JSON object containing 'medicineName', 'dosage', and 'advice' keys. If a field is missing, use an empty string "" for its value.

Transcription: "{transcription_text}"

JSON Output:"""

        logger.info(f"Sending transcription to Groq LLM for structuring (Patient: {patient_id_str})...")
        llm_response = groq_llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Recommend a powerful model for accuracy
            messages=[
                {"role": "system", "content": "You are an assistant that extracts prescription details and outputs ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 # Low temp for consistency
        )

        raw_llm_output = llm_response.choices[0].message.content
        logger.debug(f"Raw LLM output: {raw_llm_output}") # Log raw output for debugging

        # Validate JSON structure before parsing
        if not raw_llm_output or not raw_llm_output.strip().startswith('{') or not raw_llm_output.strip().endswith('}'):
             logger.error(f"LLM (Groq LLM Client) returned non-JSON structure: {raw_llm_output}")
             raise ValueError("LLM did not return valid JSON structure.")

        structured_prescription = json.loads(raw_llm_output)
        # Ensure keys exist, defaulting to empty strings
        structured_prescription.setdefault("medicineName", "")
        structured_prescription.setdefault("dosage", "")
        structured_prescription.setdefault("advice", "")

        logger.info(f"Groq LLM structured prescription: {structured_prescription}")

    except json.JSONDecodeError as e:
         logger.error(f"LLM (Groq LLM Client) returned invalid JSON: {e} - Raw: {raw_llm_output}", exc_info=True)
         return jsonify({"success": False, "message": "AI returned malformed prescription data."}), 500
    except GroqError as e:
        logger.error(f"Groq API error during structuring: {e.status_code} - {e}", exc_info=True)
        status_code = getattr(e, 'status_code', 500)
        message = f"Groq structuring error: {e}"
        if status_code == 401: message = "Groq LLM API key invalid/missing (GROQ_LLM_KEY)."
        if status_code == 429: message = "Groq LLM API rate limit reached."
        return jsonify({"success": False, "message": message}), status_code if status_code != 500 else 503
    except Exception as e:
        logger.error(f"Unexpected error structuring prescription with Groq LLM: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to structure prescription data."}), 500

    # 4. Data Validation and Saving
    try:
        medicine_name = structured_prescription.get("medicineName", "").strip()
        dosage = structured_prescription.get("dosage", "").strip()
        advice = structured_prescription.get("advice", "").strip()

        # Check if AI actually extracted something meaningful
        if not medicine_name and not dosage:
            logger.warning(f"AI failed to extract meaningful details from transcription: '{transcription_text}'")
            return jsonify({"success": False, "message": "AI could not extract prescription details. Please try speaking more clearly or enter manually."}), 400

        # Use "N/A" or "As advised" only if necessary after stripping
        final_medicine_name = medicine_name or "N/A"
        final_dosage = dosage or "As advised"
        final_advice = advice or "Follow doctor's instructions."

        record_title = f"Digital Prescription from Dr. {doctor.full_name}"
        record_description = f"Medicine: {final_medicine_name}, Dosage: {final_dosage}, Advice: {final_advice}. Generated by AI from speech (Groq). Uploaded by doctor."

        record = HealthRecord(
            user_id=user.id,
            record_type='prescription',
            title=record_title,
            description=record_description,
            file_type='text/plain',
            test_date=datetime.now().date()
        )
        db.session.add(record)
        db.session.commit()
        logger.info(f"✅ HealthRecord created (ID: {record.id}) for AI prescription (Patient: {patient_id_str}).")

        # 5. Update Conversation Memory (Optional but recommended)
        if conversation_memory: # Check if memory object exists
             prescription_data = {
                 'prescription_id': record.record_id, # Assuming record_id is generated upon commit
                 'doctor_name': doctor.full_name,
                 'medications': [{'name': final_medicine_name, 'dosage': final_dosage, 'instructions': final_advice}],
                 'diagnosis': 'N/A', # Could enhance prompt to extract this
                 'instructions': final_advice,
                 'source': 'Uploaded by doctor',
                 'upload_method': 'ai_speech_groq',
                 'ai_extracted': True,
                 'original_transcription': transcription_text
             }
             try:
                 # Ensure your memory object has this method
                 conversation_memory.add_prescription_summary(user.patient_id, prescription_data)
                 logger.info(f"Updated conversation memory for patient {patient_id_str} with AI prescription.")
             except AttributeError:
                 logger.warning("`conversation_memory` object does not have `add_prescription_summary` method.")
             except Exception as mem_e:
                  logger.error(f"Error updating conversation memory: {mem_e}", exc_info=True)
        else:
             logger.warning("Conversation memory object not found, skipping update.")


        # 6. Success Response
        return jsonify({
            "success": True,
            "message": "Prescription generated from speech successfully.",
            "structuredPrescription": { # Return the cleaned-up data
                 "medicineName": final_medicine_name,
                 "dosage": final_dosage,
                 "advice": final_advice
            },
            "recordId": record.record_id # Send back the record ID
        })

    except Exception as e:
        logger.error(f"❌ Error during saving or memory update: {e}", exc_info=True)
        db.session.rollback() # Rollback database changes on error
        return jsonify({"success": False, "message": "Failed to save the generated prescription."}), 500

# --- Add this new route ---
@app.route("/v1/patient-summary/<patient_id_str>", methods=["GET"])
def get_patient_summary_for_doctor(patient_id_str):
    """Provides a patient summary (symptoms, vitals, prescriptions) for the doctor."""
    # --- ADD LOGGING HERE ---
    logger.info(f"Accessing summary for {patient_id_str}. Session data: {dict(session)}")
    logger.info(f"Request Cookies: {request.cookies}")
    # --- END LOGGING ---
    try:
        # 1. Authenticate the Doctor
        doctor = get_current_user()
        if not doctor or not isinstance(doctor, Doctor):
            logger.warning(f"Unauthorized attempt to access patient summary for {patient_id_str}")
            return jsonify({"success": False, "message": "Doctor authentication required"}), 401

        # 2. Find the Patient
        patient = User.query.filter_by(patient_id=patient_id_str, is_active=True).first()
        if not patient:
            logger.warning(f"Patient not found for summary request: {patient_id_str}")
            return jsonify({"success": False, "message": "Patient not found"}), 404

        # 3. Gather Summary Data (Add more sophisticated logic as needed)

        # a) Recent Symptoms (Example: from last few conversation turns)
        symptoms = []
        try:
            recent_turns = ConversationTurn.query.filter_by(user_id=patient.id)\
                           .order_by(ConversationTurn.timestamp.desc())\
                           .limit(5).all() # Look at last 5 turns
            # Basic symptom extraction (enhance with NLP/AI if needed)
            symptom_keywords = ['pain', 'headache', 'fever', 'cough', 'nausea', 'dizzy', 'rash', 'ache']
            for turn in recent_turns:
                 if turn.user_message:
                     msg_lower = turn.user_message.lower()
                     for keyword in symptom_keywords:
                         if keyword in msg_lower and turn.user_message not in symptoms: # Avoid duplicates
                             symptoms.append(f"Mentioned: '{turn.user_message}' (Turn ID: {turn.id})")
                             break # Add only once per turn for simplicity
            if not symptoms:
                 symptoms.append("No specific symptoms automatically detected in recent chat.")
        except Exception as e:
            logger.error(f"Error fetching symptoms for patient {patient_id_str}: {e}", exc_info=True)
            symptoms.append("Error retrieving symptom history.")


        # b) Recent Vitals (Example: from HealthRecord in last 7 days)
        vitals = []
        try:
            seven_days_ago = datetime.now() - timedelta(days=7)
            recent_vitals_records = HealthRecord.query.filter(
                HealthRecord.user_id == patient.id,
                HealthRecord.record_type.in_(['vital', 'lab_report']), # Include relevant types
                HealthRecord.created_at >= seven_days_ago
            ).order_by(HealthRecord.created_at.desc()).limit(5).all()

            if recent_vitals_records:
                 for record in recent_vitals_records:
                      # Format nicely - customize based on how vitals are stored
                      vital_info = f"{record.title or record.record_type.capitalize()}"
                      if record.description: vital_info += f": {record.description}"
                      vital_info += f" ({record.created_at.strftime('%Y-%m-%d')})"
                      vitals.append(vital_info)
            else:
                 vitals.append("No vitals recorded in the last 7 days.")
        except Exception as e:
            logger.error(f"Error fetching vitals for patient {patient_id_str}: {e}", exc_info=True)
            vitals.append("Error retrieving vitals history.")


        # c) Active Prescriptions (Example: from HealthRecord)
        prescriptions = []
        try:
            active_prescription_records = HealthRecord.query.filter(
                HealthRecord.user_id == patient.id,
                HealthRecord.record_type == 'prescription'
                # Add logic here if prescriptions have an 'end_date' or 'status' to check activity
            ).order_by(HealthRecord.created_at.desc()).limit(5).all() # Show last 5

            if active_prescription_records:
                 for record in active_prescription_records:
                      # Extract info from description (adapt if stored differently)
                      desc = record.description or ""
                      med_info = desc.split('.')[0] # Try to get first sentence
                      if not med_info.lower().startswith("medicine:"): med_info = record.title # Fallback to title
                      prescriptions.append(f"{med_info} (from {record.created_at.strftime('%Y-%m-%d')})")
            else:
                 prescriptions.append("No active prescriptions found in record.")
        except Exception as e:
            logger.error(f"Error fetching prescriptions for patient {patient_id_str}: {e}", exc_info=True)
            prescriptions.append("Error retrieving prescription history.")


        # --- VERIFY/ADD: Ensure they are always lists before returning ---
        symptoms = symptoms if isinstance(symptoms, list) else []
        vitals = vitals if isinstance(vitals, list) else []
        prescriptions = prescriptions if isinstance(prescriptions, list) else []
        # --- END VERIFICATION ---

        # 4. Return Compiled Summary
        logger.info(f"Successfully generated summary for patient {patient_id_str} for doctor {doctor.doctor_id}")
        return jsonify({
            "success": True,
            "patientName": patient.full_name,
            "patientId": patient.patient_id,
            "symptoms": symptoms,      # Now guaranteed to be a list
            "vitals": vitals,          # Now guaranteed to be a list
            "prescriptions": prescriptions # Now guaranteed to be a list
        })

    except Exception as e:
        logger.error(f"❌ Unexpected error in get_patient_summary_for_doctor for patient {patient_id_str}: {e}", exc_info=True)
        return jsonify({"success": False, "message": "An unexpected server error occurred while fetching the summary."}), 500
    
@app.route("/v1/doctor/create-digital-prescription", methods=["POST"])
def doctor_create_digital_prescription():
    """
    Saves the final digital prescription text (potentially AI-generated and edited)
    to the patient's record. This is triggered by the 'Sign & Send' button.
    """
    try:
        doctor = get_current_user()
        if not doctor or not isinstance(doctor, Doctor):
            logger.warning("Unauthorized attempt: Doctor role required for create-digital-prescription.")
            return jsonify({"success": False, "message": "Doctor authentication required"}), 401

        data = request.get_json() or {}
        patient_id_str = data.get("patientId")
        medicine_name = data.get("medicineName", "").strip()
        dosage = data.get("dosage", "").strip()
        advice = data.get("advice", "").strip()

        if not patient_id_str or not medicine_name or not dosage:
            logger.warning(f"Missing required fields for digital prescription. PatientID: {patient_id_str}, Med: {medicine_name}, Dosage: {dosage}")
            return jsonify({"success": False, "message": "Patient ID, Medicine Name, and Dosage are required."}), 400

        user = User.query.filter_by(patient_id=patient_id_str, is_active=True).first()
        if not user:
            logger.warning(f"Patient not found for digital prescription: {patient_id_str}")
            return jsonify({"success": False, "message": "Patient not found"}), 404

        record_title = f"Digital Prescription from Dr. {doctor.full_name}"
        # Include all details in the description for clarity
        record_description = f"Medicine: {medicine_name}, Dosage: {dosage}"
        if advice:
            record_description += f", Advice: {advice}"
        record_description += ". Uploaded by doctor." # Tag the source

        record = HealthRecord(
            user_id=user.id,
            record_type='prescription',
            title=record_title,
            description=record_description,
            file_type='text/plain', # Indicates it's text-based
            test_date=datetime.now().date()
            # No image_data or file_url needed here
        )
        db.session.add(record)
        db.session.commit()
        logger.info(f"✅ Digital prescription saved by Dr. {doctor.doctor_id} for Patient {patient_id_str} (Record ID: {record.id}).")

        # Optionally add to conversation memory for consistency
        if conversation_memory:
             prescription_data = {
                 'prescription_id': record.record_id,
                 'doctor_name': doctor.full_name,
                 'medications': [{'name': medicine_name, 'dosage': dosage, 'instructions': advice}],
                 'diagnosis': 'N/A',
                 'instructions': advice,
                 'source': 'Uploaded by doctor', # Source tag
                 'upload_method': 'digital_form', # Method tag
                 'ai_extracted': False # Not directly extracted, but potentially drafted by AI initially
             }
             try:
                 conversation_memory.add_prescription_summary(user.patient_id, prescription_data)
                 logger.info(f"Updated conversation memory for digital prescription (Patient: {patient_id_str}).")
             except Exception as mem_e:
                 logger.error(f"Error updating conversation memory for digital prescription: {mem_e}", exc_info=True)

        return jsonify({
            "success": True,
            "message": "Digital prescription saved and sent to patient.",
            "recordId": record.record_id
        })

    except Exception as e:
        logger.error(f"❌ Error in doctor_create_digital_prescription: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({"success": False, "message": "Failed to save digital prescription."}), 500
    
# In chatbot.py (replace the existing doctor_upload_prescription_image function with this)

@app.route("/v1/doctor/upload-prescription", methods=["POST"])
def doctor_upload_prescription_image():
    """Handles the doctor uploading a prescription image file WITHOUT AI analysis."""
    try:
        doctor = get_current_user()
        if not doctor or not isinstance(doctor, Doctor):
            logger.warning("Unauthorized attempt: Doctor role required for upload-prescription.")
            return jsonify({"success": False, "message": "Doctor authentication required"}), 401

        data = request.get_json() or {}
        patient_id_str = data.get("patientId")
        image_data_url = data.get("imageData", "").strip() # Expecting data URL format

        if not patient_id_str or not image_data_url:
            logger.warning(f"Missing required fields for image upload. PatientID: {patient_id_str}, ImageData provided: {bool(image_data_url)}")
            return jsonify({"success": False, "message": "Patient ID and image data are required."}), 400

        user = User.query.filter_by(patient_id=patient_id_str, is_active=True).first()
        if not user:
            logger.warning(f"Patient not found for image prescription upload: {patient_id_str}")
            return jsonify({"success": False, "message": "Patient not found"}), 404

        # Extract base64 data and validate
        try:
            if "," in image_data_url:
                image_data_base64 = image_data_url.split(",", 1)[1]
            else:
                image_data_base64 = image_data_url

            base64.b64decode(image_data_base64) # Validate format
        except Exception as decode_error:
            logger.error(f"Invalid Base64 image data received: {decode_error}")
            return jsonify({"success": False, "message": "Invalid image data format."}), 400

        # --- AI Analysis Removed ---

        # Determine details for the record using only the current doctor
        record_title = f"Prescription from Dr. {doctor.full_name}"
        description = "Uploaded prescription image by doctor." # Tag source

        # Create HealthRecord
        record = HealthRecord(
            user_id=user.id,
            record_type='prescription',
            title=record_title,
            description=description,
            file_type='image/jpeg', # Assuming JPEG
            image_data=image_data_base64, # Store clean base64
            test_date=datetime.now().date()
        )

        db.session.add(record)
        db.session.commit()
        logger.info(f"✅ Doctor prescription image uploaded for Patient {patient_id_str} (Record ID: {record.id}).")

        # Optionally add basic info to conversation memory (without AI fields)
        summary_response_text = "Prescription image saved. Summary unavailable without AI analysis."
        if conversation_memory:
            prescription_data = {
                'prescription_id': record.record_id,
                'doctor_name': doctor.full_name, # Use current doctor's name
                'medications': [], # No AI extraction
                'diagnosis': '', # No AI extraction
                'instructions': '', # No AI extraction
                'source': 'Uploaded by doctor', # Source tag
                'upload_method': 'doctor_upload_image', # Method tag
                'ai_extracted': False # Mark as not AI extracted
             }
            try:
                conversation_memory.add_prescription_summary(user.patient_id, prescription_data)
                # Generate a simple confirmation, as no details were extracted
                summary_response_text = f"Dr. {doctor.full_name} uploaded a prescription image for you."
                logger.info(f"Updated conversation memory (basic info) for uploaded image (Patient: {patient_id_str}).")

            except Exception as mem_e:
                logger.error(f"Error updating conversation memory for uploaded image: {mem_e}", exc_info=True)
                summary_response_text = "Memory update failed."

        return jsonify({
            "success": True,
            "message": "Prescription image uploaded successfully. AI analysis was skipped.",
            "recordId": record.record_id,
            "extractedData": None, # Indicate no AI data
            "formatted_response": summary_response_text # Send back basic confirmation text
        })

    except Exception as e:
        logger.error(f"❌ Error in doctor_upload_prescription_image (no AI): {e}", exc_info=True)
        db.session.rollback()
        return jsonify({"success": False, "message": "Failed to upload prescription image."}), 500
       
@app.route("/v1/medicine-reminders", methods=["POST"])
def manage_medicine_reminders():
    """
    New comprehensive endpoint to manage medicine reminders (get, add, update, delete, update_adherence)
    """
    try:
        if not conversation_memory:
            return jsonify({"error": "Service unavailable."}), 503

        data = request.get_json() or {}
        user_id = data.get("userId", "").strip()
        action = data.get("action", "get").strip().lower()

        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        if action == "get":
            reminders = conversation_memory.get_medicine_reminders(user_id)
            alerts = conversation_memory.get_reminder_alerts(user_id)
            return jsonify({
                "success": True,
                "reminders": reminders,
                "pending_alerts": alerts,
                "total_reminders": len(reminders)
            })

        elif action == "add":
            medicine_data = data.get('medicine_data', {})
            conversation_memory.schedule_medicine_reminder(user_id, medicine_data)
            # --- ADD THIS LINE ---
            conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))
            return jsonify({"success": True, "message": "Reminder scheduled successfully"})

        # --- NEW: UPDATE ACTION ---
        elif action == "update":
            medicine_data = data.get('medicine_data', {})
            original_name = data.get('original_medicine_name')
            if not original_name:
                return jsonify({"error": "original_medicine_name is required for update"}), 400
            
            conversation_memory.update_medicine_reminder(user_id, original_name, medicine_data)
            # --- ADD THIS LINE ---
            conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))
            return jsonify({"success": True, "message": "Reminder updated successfully"})

        # --- NEW: DELETE ACTION ---
        elif action == "delete":
            medicine_name = data.get('medicine_name')
            if not medicine_name:
                return jsonify({"error": "medicine_name is required for delete"}), 400
            
            conversation_memory.delete_medicine_reminder(user_id, medicine_name)
            # --- ADD THIS LINE ---
            conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))
            return jsonify({"success": True, "message": "Reminder deleted successfully"})

        elif action == "update_adherence":
            medicine_name = data.get('medicine_name')
            taken_time = data.get('taken_time', datetime.now().strftime('%H:%M'))
            conversation_memory.update_reminder_adherence(user_id, medicine_name, taken_time)
            # --- ADD THIS LINE ---
            # Reschedule the reminder for the next day and save the change
            conversation_memory.recalculate_all_next_alerts(user_id)
            conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))
            return jsonify({"success": True, "message": f"Medicine {medicine_name} marked as taken"})

        else:
            return jsonify({"error": "Invalid action."}), 400
        

    except Exception as e:
        logger.error(f"Medicine reminders error: {e}", exc_info=True)
        return jsonify({"error": "Failed to manage medicine reminders"}), 500

@app.route("/v1/enhanced-sos", methods=["POST"])
def handle_enhanced_sos():
    """
    New endpoint for enhanced emergency detection and handling
    """
    try:
        data = request.get_json() or {}
        user_id = data.get("userId", "").strip()
        message = data.get("message", "").strip()
        language = data.get("language", "en")
        
        if not user_id or not message:
            return jsonify({"error": "userId and message are required"}), 400
        
        # Enhanced emergency keyword detection
        emergency_keywords = {
            'en': [
                'emergency', 'help', 'urgent', 'critical', 'severe pain', 'cannot breathe',
                'chest pain', 'heart attack', 'stroke', 'accident', 'bleeding', 'unconscious',
                'poisoning', 'burn', 'seizure', 'convulsion', 'choking', 'drowning'
            ],
            'hi': [
                'इमरजेंसी', 'मदद', 'जरूरी', 'क्रिटिकल', 'तेज दर्द', 'सांस नहीं आ रही',
                'सीने में दर्द', 'हार्ट अटैक', 'स्ट्रोक', 'एक्सीडेंट', 'खून बह रहा', 'बेहोश',
                'जहर', 'जल गया', 'मिरगी', 'दम घुट रहा', 'डूब रहा'
            ],
            'pa': [
                'ਇਮਰਜੈਂਸੀ', 'ਮਦਦ', 'ਜ਼ਰੂਰੀ', 'ਕ੍ਰਿਟੀਕਲ', 'ਤੇਜ਼ ਦਰਦ', 'ਸਾਹ ਨਹੀਂ ਆ ਰਹੀ',
                'ਛਾਤੀ ਵਿੱਚ ਦਰਦ', 'ਹਾਰਟ ਅਟੈਕ', 'ਸਟ੍ਰੋਕ', 'ਐਕਸੀਡੈਂਟ', 'ਖੂਨ ਵਗ ਰਿਹਾ', 'ਬੇਹੋਸ਼',
                'ਜ਼ਹਿਰ', 'ਸੜ ਗਿਆ', 'ਮਿਰਗੀ', 'ਦਮ ਘੁੱਟ ਰਿਹਾ', 'ਡੁੱਬ ਰਿਹਾ'
            ]
        }

        # Check for emergency keywords
        detected_keywords = []
        for keyword in emergency_keywords.get(language, emergency_keywords['en']):
            if keyword in message.lower(): # Check against lowercased message
                detected_keywords.append(keyword)

        # Critical priority indicators
        critical_indicators = [
            'cannot breathe', 'chest pain', 'heart attack', 'stroke', 'unconscious',
            'सांस नहीं आ रही', 'सीने में दर्द', 'हार्ट अटैक', 'स्ट्रोक', 'बेहोश',
            'ਸਾਹ ਨਹੀਂ ਆ ਰਹੀ', 'ਛਾਤੀ ਵਿੱਚ ਦਰਦ', 'ਹਾਰਟ ਅਟੈਕ', 'ਸਟ੍ਰੋਕ', 'ਬੇਹੋਸ਼'
        ]

        is_critical = any(indicator in message.lower() for indicator in critical_indicators) # Check against lowercased message
        needs_sos = len(detected_keywords) > 0 or is_critical

        if needs_sos:
            # Log emergency
            update_system_state('enhanced_sos', sos_triggered=1)

            # Get emergency response in user's language
            emergency_responses = {
                'en': "🚨 EMERGENCY DETECTED! I'm immediately connecting you to emergency services. Call 108 for ambulance. Stay calm, help is coming!",
                'hi': "🚨 इमरजेंसी का पता चला! मैं आपको तुरंत इमरजेंसी सेवाओं से कनेक्ट कर रहा हूं। एंबुलेंस के लिए 108 पर कॉल करें। शांत रहें, मदद आ रही है!",
                'pa': "🚨 ਇਮਰਜੈਂਸੀ ਦਾ ਪਤਾ ਲੱਗਾ! ਮੈਂ ਤੁਹਾਨੂੰ ਤੁਰੰਤ ਇਮਰਜੈਂਸੀ ਸੇਵਾਵਾਂ ਨਾਲ ਕਨੈਕਟ ਕਰ ਰਿਹਾ ਹਾਂ। ਐਂਬੂਲੈਂਸ ਲਈ 108 ਤੇ ਕਾਲ ਕਰੋ। ਸ਼ਾਂਤ ਰਹੋ, ਮਦਦ ਆ ਰਹੀ ਹੈ!"
            }

            response_text = emergency_responses.get(language, emergency_responses['en'])

            return jsonify({
                "success": True,
                "emergency_detected": True,
                "critical": is_critical,
                "detected_keywords": detected_keywords,
                "response": response_text,
                "action": "TRIGGER_SOS",
                "parameters": {
                    "emergency_number": "108",
                    "type": "medical_emergency",
                    "auto_dial": is_critical,  # Auto-dial for critical cases
                    "priority": "high" if is_critical else "medium"
                }
            })
        else:
            return jsonify({
                "success": True,
                "emergency_detected": False,
                "response": "I understand you need help. Can you tell me more about your situation?",
                "action": "CONTINUE_SYMPTOM_CHECK",
                "parameters": {}
            })

    except Exception as e:
        logger.error(f"Enhanced SOS detection error: {e}")
        return jsonify({"error": "Failed to process emergency detection"}), 500

@app.route("/v1/test-prescription", methods=["GET"])
def test_prescription_endpoint():
    """Test endpoint to verify prescription API is working"""
    return jsonify({
        "success": True,
        "message": "Prescription API is working",
        "available_endpoints": [
            "POST /v1/prescription-summary",
            "POST /v1/medicine-reminders",
            "POST /v1/upload-prescription"
        ]
    })

        

@app.route("/v1/prescription-summary", methods=["POST"])
def get_prescription_summary():
    """Get prescription summary for user"""
    try:
        logger.info("Prescription summary endpoint called")
        data = request.get_json() or {}
        user_id = data.get("userId", "").strip()
        prescription_id = data.get("prescriptionId")  # Optional, gets latest if not provided

        logger.info(f"Prescription summary request: user_id={user_id}, prescription_id={prescription_id}")

        if not user_id:
            logger.error("Prescription summary: userId is required")
            return jsonify({"error": "userId is required"}), 400

        # Get prescription summary from conversation memory
        if hasattr(initialize_ai_components, '_conversation_memory'):
            prescription_data = initialize_ai_components._conversation_memory.get_prescription_summary(user_id, prescription_id)
        else:
            prescription_data = None

        logger.info(f"Found prescription data: {bool(prescription_data)}")

        if not prescription_data:
            logger.warning(f"No prescription found for user {user_id}")
            return jsonify({
                "success": False,
                "message": "No prescription found for this user"
            }), 404

        # Generate summary response using response generator
        if response_generator:
            summary_response = response_generator.generate_prescription_summary_response(
                prescription_data, 'en'  # Default to English for prescription display
            )
        else:
            summary_response = "Prescription summary not available."

        logger.info(f"Prescription summary generated successfully for user {user_id}")
        return jsonify({
            "success": True,
            "prescription_summary": prescription_data,
            "formatted_response": summary_response
        })

    except Exception as e:
        logger.error(f"Prescription summary error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to get prescription summary"}), 500


def _get_or_create_booking_context(user_id: str) -> Dict[str, Any]:
    """Retrieves or starts a new booking task context from memory."""
    if hasattr(initialize_ai_components, '_conversation_memory'):
        task = initialize_ai_components._conversation_memory.get_current_task(user_id)
        # If a booking is already in progress, use its context
        if task.get('task') == 'appointment_booking' and task.get('context'):
            return task.get('context')

        # Otherwise, start a new booking task with an empty context
        new_context = {}
        initialize_ai_components._conversation_memory.set_current_task(user_id, 'appointment_booking', new_context)
        return new_context
    return {}

def _get_available_specialties_from_db() -> list:
    """Gets a unique, sorted list of available doctor specialties from the database."""
    try:
        # Query for distinct, non-null, non-empty specializations from active doctors
        specialties = db.session.query(Doctor.specialization).filter(
            Doctor.is_active==True,
            Doctor.specialization.isnot(None),
            Doctor.specialization != ''
        ).distinct().all()
        # Flatten the list of tuples and sort it
        return sorted([s[0] for s in specialties])
    except Exception as e:
        logger.error(f"Error getting specialties from database: {e}")
        # Fallback to default specialties if the database query fails
        return ["General Physician", "Dermatologist", "Child Specialist"]

def _get_doctors_for_specialty(specialty: str) -> list:
    """Gets a list of active doctors for a specific specialty from the database."""
    try:
        # Use ilike for case-insensitive matching
        doctors = Doctor.query.filter(
            Doctor.specialization.ilike(f'%{specialty}%'),
            Doctor.is_active == True
        ).all()
        return doctors
    except Exception as e:
        logger.error(f"Error getting doctors for specialty {specialty}: {e}")
        return []

def _get_available_time_slots() -> list:
    """Gets available time slots, falling back to defaults."""
    # This function can be expanded to check doctor schedules in the future
    return [
        {"text": "10:00 AM", "iso": "10:00"},
        {"text": "11:30 AM", "iso": "11:30"},
        {"text": "02:00 PM", "iso": "14:00"},
        {"text": "04:30 PM", "iso": "16:30"}
    ]

# In chatbot.py, replace the /v1/predict function with this final version:

# In chatbot.py, replace your entire /v1/predict function with this one
# In chatbot.py, replace the existing function
def _get_or_create_symptom_context(user_id: str, initial_symptom: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves or starts a new symptom checker context, storing symptoms."""
    if hasattr(initialize_ai_components, '_conversation_memory'):
        # Get current task from conversation memory
        current_task = initialize_ai_components._conversation_memory.get_current_task(user_id)

        if current_task.get('task') == 'symptom_triage' and current_task.get('context'):
            context = current_task.get('context')
            # Ensure symptoms_reported list exists
            if 'symptoms_reported' not in context:
                context['symptoms_reported'] = []
            return context

        # Start a new task if one isn't active
        new_context = {
            'turn_count': 0,
            'symptoms_reported': [] # <-- New: A list to store user's answers
        }
        if initial_symptom:
            new_context['symptoms_reported'].append(initial_symptom)

        initialize_ai_components._conversation_memory.set_current_task(user_id, 'symptom_triage', new_context)
        return new_context
    return {'turn_count': 0, 'symptoms_reported': []}

# In chatbot.py
# In chatbot.py

# In chatbot.py

@app.route("/v1/predict", methods=["POST"])
def predict():
    """
    Enhanced prediction endpoint with flexible state management and isolated, stateful logic
    for booking and symptom checking.
    """
    try:
        start_time = time.time()
        update_system_state('predict')

        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        user_id_str = (data.get("userId") or "").strip()
        button_action = data.get("buttonAction")
        button_params = data.get("parameters", {})

        if not user_id_str or not user_message:
            return jsonify({"error": "userId and message are required."}), 400

        current_user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found.", "login_required": True}), 401

        # --- START: MODIFIED STATE MANAGEMENT LOGIC ---

        # Get current task from conversation memory
        if hasattr(initialize_ai_components, '_conversation_memory'):
            current_task_from_memory = initialize_ai_components._conversation_memory.get_current_task(current_user.patient_id)
            task_in_progress = current_task_from_memory.get('task') if current_task_from_memory else None
        else:
            task_in_progress = None

        nlu_understanding = nlu_processor.understand_user_intent(user_message, sehat_sahara_mode=True)
        
        # --- FIX 1: Persist Detected Language ---
        # This addresses Root Cause Analysis Problem 1
        detected_lang = nlu_understanding.get('language_detected', 'hi')
        if current_user and current_user.preferred_language != detected_lang:
            logger.info(f"Updating user {current_user.patient_id} preferred language from '{current_user.preferred_language}' to '{detected_lang}'")
            current_user.preferred_language = detected_lang
            db.session.add(current_user) # Add to session, will be committed with the ConversationTurn
        # --- END OF FIX 1 ---
        
        primary_intent = nlu_understanding.get('primary_intent', 'general_inquiry')

        # FLEXIBLE STATE MANAGEMENT: Check if the user's new intent is unrelated to the current task.
        related_booking_intents = ['appointment_booking', 'SELECT_SPECIALTY', 'SELECT_DOCTOR', 'SELECT_DATE', 'SELECT_TIME', 'SELECT_MODE']
        related_symptom_intents = ['symptom_triage', 'CONTINUE_SYMPTOM_CHECK']

        # Special handling for symptom triage context (from previous fix)
        if task_in_progress == 'symptom_triage' and primary_intent == 'out_of_scope':
            symptom_followup_indicators = [
                'days', 'hours', 'weeks', 'since', 'ago', 'started', 'began',
                'pain', 'hurt', 'ache', 'feel', 'feeling', 'still', 'now', 'when',
                'morning', 'night', 'yesterday', 'today', 'severe', 'mild', 'moderate'
            ]
            message_words = user_message.lower().split()
            is_likely_symptom_followup = any(indicator in message_words for indicator in symptom_followup_indicators)
            if is_likely_symptom_followup:
                logger.info(f"Correcting misclassified symptom follow-up for user {current_user.patient_id}")
                primary_intent = 'symptom_triage'

        is_unrelated_intent = (
            task_in_progress and
            primary_intent not in related_booking_intents and
            button_action not in related_booking_intents and
            primary_intent not in related_symptom_intents and
            button_action not in related_symptom_intents and
            primary_intent != task_in_progress
        )

        if is_unrelated_intent:
            logger.warning(f"User {current_user.patient_id} switched intent from '{task_in_progress}' to '{primary_intent}'. Completing old task.")
            if hasattr(initialize_ai_components, '_conversation_memory'):
                initialize_ai_components._conversation_memory.complete_task(current_user.patient_id)
            task_in_progress = None # Reset the task so the logic below can start a new one.

        # --- END: MODIFIED STATE MANAGEMENT LOGIC ---

        ai_message_override = user_message

        # --- TASK MANAGEMENT LOGIC ---
        if task_in_progress == 'appointment_booking':
            # --- START: ADDED RESTART CHECK ---
            if primary_intent == 'appointment_booking' and not button_action:
                # User explicitly said "book appointment" again, restart the flow
                logger.info(f"User {current_user.patient_id} requested to restart booking flow.")
                booking_context = _get_or_create_booking_context(current_user.patient_id)
                booking_context.clear() # Clear old context
                booking_context['last_step'] = 'ask_specialty'
                available_specialties = _get_available_specialties_from_db()
                specialty_buttons = ', '.join([f'{{"text": "{s}", "action": "SELECT_SPECIALTY", "parameters": {{"specialty": "{s}"}}}}' for s in available_specialties])
                ai_message_override = f"CONTEXT: User wants to restart booking. Ask them to select a specialty using these buttons: [{specialty_buttons}]"
                if hasattr(initialize_ai_components, '_conversation_memory'):
                    initialize_ai_components._conversation_memory.set_current_task(current_user.patient_id, 'appointment_booking', booking_context)
                # Skip the rest of the old booking logic for this turn
                task_in_progress = 'handled_restart' # Set a flag to prevent further processing in this block
            # --- END: ADDED RESTART CHECK ---

            # --- Original booking logic continues below, now conditional ---
            if task_in_progress == 'appointment_booking': # Check flag
                booking_context = _get_or_create_booking_context(current_user.patient_id)
                last_step = booking_context.get('last_step')
                logger.info(f"Booking flow active for user {current_user.patient_id}: step={last_step}")

                if last_step == 'ask_specialty':
                    selected_specialty = button_params.get('specialty') if button_action == 'SELECT_SPECIALTY' else user_message
                    doctors = _get_doctors_for_specialty(selected_specialty)
                    if doctors:
                        booking_context['specialty'] = selected_specialty
                        booking_context['last_step'] = 'ask_doctor'
                        doctor_buttons = ', '.join([f'{{"text": "Dr. {doc.full_name}", "action": "SELECT_DOCTOR", "parameters": {{"doctor_id": "{doc.doctor_id}"}}}}' for doc in doctors[:5]])
                        ai_message_override = f"CONTEXT: The user chose '{selected_specialty}'. Ask them to choose a doctor using these buttons: [{doctor_buttons}]"
                    else:
                        ai_message_override = f"CONTEXT: No doctors found for '{selected_specialty}'. Apologize and restart by asking for a specialty again."
                        booking_context.clear()

                elif last_step == 'ask_doctor':
                    selected_doctor_id = button_params.get('doctor_id')
                    doctor = Doctor.query.filter_by(doctor_id=selected_doctor_id).first() if selected_doctor_id else None
                    if not doctor:
                        doctor_name_match = re.search(r'(?:Dr\.\s*)?(\w+\s+\w+)', user_message)
                        if doctor_name_match:
                            doctor_name = doctor_name_match.group(1)
                            doctor = Doctor.query.filter(Doctor.full_name.ilike(f'%{doctor_name}%')).first()
                    if doctor:
                        booking_context['doctor_id'] = doctor.doctor_id
                        booking_context['doctor_name'] = doctor.full_name
                        booking_context['last_step'] = 'ask_date'
                        dates = [{"text": (datetime.now() + timedelta(days=i)).strftime('%a, %b %d'), "iso": (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')} for i in range(5)]
                        dates[0]['text'] = 'Today'
                        dates[1]['text'] = 'Tomorrow'
                        date_buttons = ', '.join([f'{{"text": "{d["text"]}", "action": "SELECT_DATE", "parameters": {{"dateISO": "{d["iso"]}"}}}}' for d in dates])
                        ai_message_override = f"CONTEXT: User chose Dr. {doctor.full_name}. Ask for a date with these buttons: [{date_buttons}]"
                    else:
                        ai_message_override = "CONTEXT: Invalid doctor. Please choose a doctor from the list."

                elif last_step == 'ask_date' and button_action == 'SELECT_DATE':
                    selected_date_iso = button_params.get('dateISO')
                    if selected_date_iso:
                        booking_context['date'] = selected_date_iso
                        booking_context['last_step'] = 'ask_time'
                        time_slots = _get_available_time_slots()
                        time_buttons = ', '.join([f'{{"text": "{s["text"]}", "action": "SELECT_TIME", "parameters": {{"timeISO": "{s["iso"]}"}}}}' for s in time_slots])
                        ai_message_override = f"CONTEXT: User chose a date. Ask for a time using these buttons: [{time_buttons}]"
                    else:
                        ai_message_override = "CONTEXT: Invalid date. Please pick a date from the list."

                elif last_step == 'ask_time' and button_action == 'SELECT_TIME':
                    selected_time_iso = button_params.get('timeISO')
                    if selected_time_iso:
                        booking_context['time'] = selected_time_iso
                        booking_context['last_step'] = 'ask_mode'
                        modes = ["Video Call", "Audio Call", "Photo-based"]
                        mode_buttons = ', '.join([f'{{"text": "{mode}", "action": "SELECT_MODE", "parameters": {{"mode": "{mode}"}}}}' for mode in modes])
                        ai_message_override = f"CONTEXT: User chose a time. Ask for the mode with these buttons: [{mode_buttons}]"
                    else:
                        ai_message_override = "CONTEXT: Invalid time. Please pick a time from the list."

                elif last_step == 'ask_mode' and button_action == 'SELECT_MODE':
                    selected_mode = button_params.get('mode')
                    if selected_mode in ["Video Call", "Audio Call", "Photo-based"]:
                        booking_context['mode'] = selected_mode
                        booking_context['last_step'] = 'finalize'
                        ai_message_override = "CONTEXT: All details selected. Your action MUST be 'FINALIZE_BOOKING'. Confirm with the user."
                    else:
                        ai_message_override = "CONTEXT: Invalid mode. Please pick a mode from the list."
                else:
                    # This handles cases where the input doesn't match the expected step (e.g., "my head pains")
                    logger.warning(f"Booking flow stalled. User input '{user_message}' did not match step '{last_step}'. Re-prompting.")
                    ai_message_override = f"CONTEXT: User provided an unexpected response. Please re-ask them to select an option for the current step: '{last_step}'."

                # Save context only if the restart wasn't just handled
                if hasattr(initialize_ai_components, '_conversation_memory') and task_in_progress != 'handled_restart':
                     initialize_ai_components._conversation_memory.set_current_task(current_user.patient_id, 'appointment_booking', booking_context)

        # --- START: REVISED SYMPTOM TRIAGE LOGIC ---
        elif task_in_progress == 'symptom_triage':
            symptom_context = _get_or_create_symptom_context(current_user.patient_id)

            # Always treat follow-up messages as part of the symptom check
            primary_intent = 'symptom_triage'

            symptom_context['symptoms_reported'].append(user_message)
            turn_count = symptom_context.get('turn_count', 0)

            if turn_count < 3:
                symptoms_history = "; ".join(symptom_context['symptoms_reported'])
                ai_message_override = (
                    f"CONTEXT: This is a symptom check conversation. The user has already provided this information: '{symptoms_history}'. "
                    f"Their latest reply is: '{user_message}'. "
                    "Acknowledge their latest reply and ask the *next* logical clarifying question. "
                    "DO NOT repeat a question that has already been answered in the history."
                )

                # Correctly increment the turn counter
                symptom_context['turn_count'] = turn_count + 1

                if hasattr(initialize_ai_components, '_conversation_memory'):
                    initialize_ai_components._conversation_memory.set_current_task(current_user.patient_id, 'symptom_triage', symptom_context)

            else:
                reported_symptoms_str = '; '.join(symptom_context['symptoms_reported'])
                ai_message_override = (
                    f"CONTEXT: The user has reported the following symptoms: '{reported_symptoms_str}'. "
                    "You have asked enough questions. Provide a simple, safe home remedy for these symptoms. "
                    "Then, you MUST include this exact disclaimer: 'This is not medical advice. For a proper diagnosis, please consult a doctor.' "
                    "After the remedy and disclaimer, provide guidance on how to use the medicine scan and prescription upload features, "
                    "then show these interactive buttons for navigation: "
                    "[{\"text\": \"📷 Scan Medicine\", \"action\": \"START_MEDICINE_SCANNER\", \"parameters\": {}, \"style\": \"primary\"}, "
                    "{\"text\": \"📤 Upload Prescription\", \"action\": \"UPLOAD_PRESCRIPTION\", \"parameters\": {}, \"style\": \"secondary\"}]. "
                    "Your final action should be 'SHOW_MEDICINE_REMEDY'."
                    )
                if hasattr(initialize_ai_components, '_conversation_memory'):
                    initialize_ai_components._conversation_memory.complete_task(current_user.patient_id)
        # --- END: REVISED SYMPTOM TRIAGE LOGIC ---

        # --- This 'else' block handles starting NEW tasks ---
        else:
            if primary_intent == 'appointment_booking':
                booking_context = _get_or_create_booking_context(current_user.patient_id)
                booking_context.clear()
                booking_context['last_step'] = 'ask_specialty'
                available_specialties = _get_available_specialties_from_db()
                specialty_buttons = ', '.join([f'{{"text": "{s}", "action": "SELECT_SPECIALTY", "parameters": {{"specialty": "{s}"}}}}' for s in available_specialties])
                ai_message_override = f"CONTEXT: User wants to book an appointment. Ask them to select a specialty using these buttons: [{specialty_buttons}]"
                if hasattr(initialize_ai_components, '_conversation_memory'):
                    initialize_ai_components._conversation_memory.set_current_task(current_user.patient_id, 'appointment_booking', booking_context)

            elif primary_intent == 'symptom_triage':
                symptom_context = _get_or_create_symptom_context(current_user.patient_id, initial_symptom=user_message)
                # Ensure context is clean for a new triage
                symptom_context.clear()
                symptom_context['symptoms_reported'] = [user_message] # Start with the first symptom
                symptom_context['turn_count'] = 1 # Set turn count to 1
                ai_message_override = f"CONTEXT: Start of a symptom check. User said '{user_message}'. Acknowledge their symptom and ask your first clarifying question (e.g., 'For how long?' or 'Is it a sharp or dull pain?')."
                if hasattr(initialize_ai_components, '_conversation_memory'):
                    initialize_ai_components._conversation_memory.set_current_task(current_user.patient_id, 'symptom_triage', symptom_context)

            # --- Guidance Buttons Logic (No changes needed here) ---
            elif primary_intent in ['medicine_scan', 'how_to_medicine_scan']:
                if hasattr(initialize_ai_components, '_conversation_memory'):
                    initialize_ai_components._conversation_memory.complete_task(current_user.patient_id) # Clear any previous task
                ai_message_override = (
                    "CONTEXT: The user wants to scan a medicine. First, provide a brief, friendly guide on how to use the scanner (e.g., 'I can help with that! Just point your camera at the medicine...'). "
                    "Then, your action MUST be 'SHOW_GUIDANCE' and you MUST include this exact button in the interactive_buttons array: "
                    '[{"text": "📷 Open Scanner", "action": "START_MEDICINE_SCANNER", "parameters": {}, "style": "primary"}]'
                )

            elif primary_intent in ['prescription_upload', 'how_to_prescription_upload', 'prescription_inquiry']:
                if hasattr(initialize_ai_components, '_conversation_memory'):
                    initialize_ai_components._conversation_memory.complete_task(current_user.patient_id) # Clear any previous task
                ai_message_override = (
                    "CONTEXT: The user wants to upload a prescription. First, provide a brief, friendly guide on how to upload (e.g., 'Okay, let\\'s upload your prescription. Make sure the photo is clear...'). "
                    "Then, your action MUST be 'SHOW_GUIDANCE' and you MUST include this exact button in the interactive_buttons array: "
                    '[{"text": "📤 Upload Prescription", "action": "UPLOAD_PRESCRIPTION", "parameters": {}, "style": "primary"}]'
                )
            # --- End Guidance Buttons Logic ---

        # --- AI RESPONSE GENERATION ---
        if hasattr(initialize_ai_components, '_conversation_memory'):
            history = initialize_ai_components._conversation_memory.get_conversation_context(current_user.patient_id, turns=8)
        else:
            history = []
            
        # --- FIX 2: Pass Detected Language to AI ---
        # This addresses Root Cause Analysis Problem 2
        # We use the 'detected_lang' variable from Fix 1
        context = {
            "user_intent": primary_intent, 
            "context_history": history,
            "language": detected_lang 
        }
        # --- END OF FIX 2 ---

        action_payload = None
        if sehat_sahara_client and sehat_sahara_client.is_available:
            action_payload_str = sehat_sahara_client.generate_sehatsahara_response(
                user_message=ai_message_override, context=context
            )
            if action_payload_str:
                try:
                    # Clean the JSON string (remove potential markdown, leading/trailing quotes)
                    cleaned_response = action_payload_str.strip().replace('```json', '').replace('```', '').strip()
                    if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
                        cleaned_response = cleaned_response[1:-1]

                    action_payload = json.loads(cleaned_response)

                    # --- Final Booking Handling (No changes needed here) ---
                    if action_payload.get("action") == "FINALIZE_BOOKING":
                        booking_context = _get_or_create_booking_context(current_user.patient_id)
                        doc_id_str = booking_context.get('doctor_id') # Get the string ID like 'DOC002'
                        appt_date = booking_context.get('date')
                        appt_time_str = booking_context.get('time')
                        # Find doctor using the string ID
                        doctor = Doctor.query.filter_by(doctor_id=doc_id_str).first() if doc_id_str else None


                        if doctor and appt_date and appt_time_str:
                            appt_datetime = datetime.fromisoformat(f"{appt_date}T{appt_time_str}:00")

                            action_payload['response'] = f"Great! I'm confirming your appointment with Dr. {doctor.full_name} for {appt_datetime.strftime('%A, %b %d at %I:%M %p')}. One moment..."
                            action_payload['action'] = 'EXECUTE_BOOKING'
                            action_payload['parameters'] = {
                                'doctorId': doctor.id, # Use the integer primary key ID for the booking endpoint
                                'appointmentDatetime': appt_datetime.isoformat(),
                                'appointmentType': booking_context.get('mode', 'Video Call'),
                                'chiefComplaint': f"Consultation for {booking_context.get('specialty', 'Unknown')}"
                            }
                            action_payload['interactive_buttons'] = []

                            if hasattr(initialize_ai_components, '_conversation_memory'):
                                initialize_ai_components._conversation_memory.complete_task(current_user.patient_id)
                        else:
                            logger.error(f"Missing booking details for FINALIZE_BOOKING: doctor={doctor}, date={appt_date}, time={appt_time_str}")
                            action_payload['response'] = "I'm sorry, there was an error retrieving the booking details. Let's start over."
                            action_payload['action'] = 'BOOKING_FAILED'
                            if hasattr(initialize_ai_components, '_conversation_memory'):
                                initialize_ai_components._conversation_memory.complete_task(current_user.patient_id)
                    # --- End Final Booking Handling ---

                    # --- Medicine Remedy Buttons (No changes needed here) ---
                    if action_payload.get("action") == "SHOW_MEDICINE_REMEDY" and "interactive_buttons" not in action_payload:
                        action_payload["interactive_buttons"] = [
                            {"text": "📷 Scan Medicine", "action": "START_MEDICINE_SCANNER", "parameters": {}, "style": "primary"},
                            {"text": "📤 Upload Prescription", "action": "UPLOAD_PRESCRIPTION", "parameters": {}, "style": "secondary"}
                        ]
                    # --- End Medicine Remedy Buttons ---

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse AI response as JSON: {e}")
                    logger.warning(f"Raw AI response trying to parse: {cleaned_response}")
                    action_payload = None
                except Exception as e:
                    logger.error(f"Error processing AI response: {e}", exc_info=True)
                    action_payload = None

        # --- Fallback Response Generation (No changes needed) ---
        if not action_payload:
            logger.warning(f"AI failed or unavailable. Using local fallback for intent: {primary_intent}")
            action_payload = response_generator.generate_response(
                user_message=user_message,
                nlu_result=nlu_understanding,
                user_context={'user_id': current_user.patient_id},
                conversation_history=history
            )
            # Ensure fallback has necessary keys
            if "action" not in action_payload: action_payload["action"] = "CONTINUE_CONVERSATION"
            if "parameters" not in action_payload: action_payload["parameters"] = {}
            if "interactive_buttons" not in action_payload: action_payload["interactive_buttons"] = []


        # --- FINAL PROCESSING & SAVING ---
        # Use primary_intent determined *after* state management for saving
        action_payload_str = json.dumps(action_payload, ensure_ascii=False)
        turn_record = ConversationTurn(
            user_id=current_user.id,
            user_message=user_message,
            bot_response=action_payload_str,
            detected_intent=primary_intent, # Use the potentially corrected intent
            action_triggered=action_payload.get('action')
        )
        db.session.add(turn_record)
        db.session.commit()

        # Update button visibility if needed (No changes needed here)
        if action_payload.get('action') == 'SHOW_MEDICINE_REMEDY':
            if hasattr(initialize_ai_components, '_conversation_memory'):
                initialize_ai_components._conversation_memory.update_button_visibility(current_user.patient_id, 'medicine_recommendation')

        # Save conversation memory (No changes needed here)
        try:
            if hasattr(initialize_ai_components, '_conversation_memory'):
                initialize_ai_components._conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))
        except Exception as save_error:
            logger.warning(f"Failed to save conversation memory: {save_error}")

        response_time = time.time() - start_time
        logger.info(f"User {current_user.patient_id} processed in {response_time:.2f}s. Intent: {primary_intent}, Action: {action_payload.get('action')}")

        return jsonify(action_payload)

    # --- Exception Handling (No changes needed) ---
    except Exception as e:
        logger.error(f"FATAL ERROR in /predict endpoint: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({"response": "I'm having a technical issue. Please try again.", "action": "SHOW_APP_FEATURES", "interactive_buttons": []}), 500

# In chatbot.py, replace the whole book_doctor function with this one
@app.route("/v1/book-doctor", methods=["POST"])
def book_doctor():
    try:
        update_system_state('book_doctor')
        data = request.get_json() or {}

        # 1. Authenticate User (this part is correct)
        user = get_current_user()
        if not user:
            user_id_param = data.get("userId")
            if user_id_param:
                user = User.query.filter_by(patient_id=user_id_param).first()
        if not user:
            return jsonify({"success": False, "message": "Authentication required or user not found"}), 401

        doctor_id_val = data.get("doctorId")
        if doctor_id_val is None:
             return jsonify({"success": False, "message": "doctorId is required"}), 400

        # --- START OF FIX: ROBUST DOCTOR LOOKUP ---
        # Use a direct primary key lookup, which is fast and reliable.
        # This is safer than the previous OR query.
        try:
            doctor = Doctor.query.get(int(doctor_id_val))
        except (ValueError, TypeError):
            doctor = None
        # --- END OF FIX ---

        if not doctor:
            logger.warning(f"Doctor not found for ID: {doctor_id_val}")
            return jsonify({"success": False, "message": "Doctor not found"}), 404

        appointment_dt = (data.get("appointmentDatetime") or "").strip()
        appointment_type = (data.get("appointmentType") or "consultation").strip()
        chief_complaint = (data.get("chiefComplaint") or "").strip()
        symptoms = data.get("symptoms") or []

        if not appointment_dt:
            return jsonify({"success": False, "message": "appointmentDatetime is required"}), 400

        try:
            when = datetime.fromisoformat(appointment_dt)
        except Exception:
            return jsonify({"success": False, "message": "Invalid appointmentDatetime. Use ISO 8601 format."}), 400

        appt = Appointment(
            user_id=user.id,
            doctor_id=doctor.id,
            appointment_datetime=when,
            appointment_type=appointment_type,
            chief_complaint=chief_complaint
        )
        appt.set_symptoms(symptoms)
        db.session.add(appt)

        try:
            session_record_id = session.get('session_record_id')
            if session_record_id:
                s = UserSession.query.get(session_record_id)
                if s:
                    s.appointments_booked_in_session += 1
        except Exception as session_error:
            logger.error(f"Failed to update session counter: {session_error}")
            pass

        db.session.commit()
        update_system_state('book_doctor', appointments_booked=1)

        return jsonify({
            "success": True,
            "message": "Appointment created.",
            "appointment": {
                "appointmentId": appt.appointment_id,
                "doctor": {"id": doctor.doctor_id, "name": doctor.full_name, "specialization": doctor.specialization},
                "datetime": appt.appointment_datetime.isoformat() + "Z",
                "type": appt.appointment_type,
                "status": appt.status
            }
        })

    except Exception as e:
        logger.error(f"Book doctor error: {e}", exc_info=True)
        update_system_state('book_doctor', success=False)
        db.session.rollback()
        return jsonify({"success": False, "message": "Failed to book appointment due to a server error."}), 500

@app.route("/v1/complete-appointment", methods=["POST"])
def complete_appointment():
    """
    New endpoint to mark appointment as completed and set post-appointment feedback flag
    """
    try:
        update_system_state('complete_appointment')
        
        data = request.get_json() or {}
        user_id = data.get("userId", "").strip()
        appointment_id = data.get("appointmentId", "").strip()
        
        if not user_id or not appointment_id:
            return jsonify({"success": False, "message": "userId and appointmentId are required"}), 400
        
        # Find appointment
        appointment = Appointment.query.filter_by(appointment_id=appointment_id).first()
        if not appointment:
            return jsonify({"success": False, "message": "Appointment not found"}), 404
        
        # Mark as completed
        appointment.complete_appointment()
        db.session.commit()
        
        if hasattr(initialize_ai_components, '_conversation_memory'):
            initialize_ai_components._conversation_memory.update_appointment_status(
                user_id=user_id,
                appointment_id=appointment_id,
                status='completed',
                appointment_date=appointment.appointment_datetime
            )
        
        logger.info(f"✅ Appointment {appointment_id} marked as completed for user {user_id}")
        
        return jsonify({
            "success": True,
            "message": "Appointment marked as completed",
            "feedback_pending": True
        })
        
    except Exception as e:
        logger.error(f"Complete appointment error: {e}")
        return jsonify({"success": False, "message": "Failed to complete appointment"}), 500
# Add this entire new function to chatbot.py

@app.route("/v1/scan-medicine", methods=["POST"])
def scan_medicine():
    """
    Handles medicine scanning from an image using AI analysis.
    """
    try:
        update_system_state('scan_medicine')
        data = request.get_json() or {}
        image_data = data.get("imageData")
        user_message = data.get("message", "Please identify this medicine from the image.")

        if not image_data:
            return jsonify({"success": False, "error": "No image data was provided."}), 400

        # Check if the AI client for image analysis is available
        if groq_scout and groq_scout.is_available:
            # Clean the "data:image/jpeg;base64," prefix from the image data string
            if "," in image_data:
                image_data_clean = image_data.split(",", 1)[1]
            else:
                image_data_clean = image_data

            # Call the AI to interpret the image
            analysis_result = groq_scout.interpret_medicine_image(
                user_message=user_message,
                image_b64=image_data_clean,
                language="en"
            )

            if analysis_result:
                return jsonify({
                    "success": True,
                    "medicine_info": analysis_result
                })
            else:
                return jsonify({"success": False, "error": "AI analysis failed to identify the medicine."}), 500
        else:
            return jsonify({"success": False, "error": "The image analysis service is currently unavailable."}), 503

    except Exception as e:
        logger.error(f"❌ Scan medicine error: {e}", exc_info=True)
        update_system_state('scan_medicine', success=False)
        return jsonify({"success": False, "error": "An internal server error occurred during the scan."}), 500
        
@app.route("/v1/history", methods=["POST"])
def get_history():
    """Get comprehensive conversation history with analytics"""
    try:
        update_system_state('get_history')
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        user_id_str = data.get("userId", "")
        limit = min(data.get("limit", 50), 100)
        include_analysis = data.get("includeAnalysis", False)

        if not user_id_str:
            return jsonify({"error": "User ID is required"}), 400

        current_user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found"}), 401

        # Get conversation turns with ordering
        turns = ConversationTurn.query.filter_by(user_id=current_user.id)\
                .order_by(ConversationTurn.timestamp.asc())\
                .limit(limit).all()

        # Format conversation history
        chat_log = []
        for turn in turns:
            # User message
            user_entry = {
                "role": "user",
                "content": turn.user_message,
                "timestamp": turn.timestamp.isoformat(),
                "turn_id": turn.id
            }

            # Assistant response with optional analysis
            assistant_entry = {
                "role": "assistant",
                "content": turn.bot_response,
                "timestamp": turn.timestamp.isoformat(),
                "turn_id": turn.id
            }

            if include_analysis:
                assistant_entry["analysis"] = {
                    "intent": turn.detected_intent,
                    "confidence": turn.intent_confidence,
                    "language": turn.language_detected,
                    "urgency": turn.urgency_level,
                    "action": turn.action_triggered,
                    "action_parameters": turn.get_action_parameters(),
                    "context_entities": turn.get_context_entities()
                }

            chat_log.extend([user_entry, assistant_entry])

        # Get user progress summary
        if hasattr(initialize_ai_components, '_conversation_memory'):
            user_summary = initialize_ai_components._conversation_memory.get_user_summary(user_id_str)
            conversation_stage = initialize_ai_components._conversation_memory.get_conversation_stage_db(current_user.patient_id)
        else:
            user_summary = {}
            conversation_stage = 'general'

        response_data = {
            "success": True,
            "history": chat_log,
            "summary": {
                "total_conversations": current_user.total_conversations,
                "current_stage": conversation_stage,
                "risk_level": getattr(current_user, 'current_risk_level', 'low'),
                "improvement_trend": getattr(current_user, 'improvement_trend', 'stable'),
                "member_since": current_user.created_at.isoformat(),
                "last_interaction": current_user.last_login.isoformat() if current_user.last_login else None
            }
        }

        # Add detailed progress if available
        if user_summary.get('exists'):
            response_data["progress_analytics"] = user_summary.get('progress_metrics', {})
            response_data["method_analytics"] = user_summary.get('method_effectiveness', {})
            response_data["risk_assessment"] = user_summary.get('risk_assessment', {})

        logger.info(f"✅ History retrieved for {user_id_str}: {len(turns)} turns")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ History retrieval error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('get_history', success=False)

        return jsonify({
            "error": "Failed to retrieve conversation history",
            "message": "Please try again later"
        }), 500

@app.route("/v1/user-stats", methods=["POST"])
def get_user_stats():
    try:
        update_system_state('get_user_stats')
        data = request.get_json() or {}
        user_id_str = (data.get("userId") or "").strip()
        if not user_id_str:
            return jsonify({"error": "User ID is required"}), 400
        current_user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
        if not current_user:
            return jsonify({"error": "User not found"}), 401

        stats = get_user_statistics(current_user.id) or {}
        return jsonify({"success": True, **stats})
    except Exception as e:
        logger.error(f"User stats error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('get_user_stats', success=False)
        return jsonify({"error": "Failed to retrieve user statistics"}), 500

@app.route("/v1/health", methods=["GET"])
def health_check():
    """Comprehensive system health check with Ollama status"""
    try:
        # System health overview
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": round((datetime.now() - system_state['startup_time']).total_seconds() / 3600, 2),
            "components": system_status,
            "api_ollama_integration": {
                "available": sehat_sahara_client.is_available,
                "status": "connected" if sehat_sahara_client.is_available else "fallback_mode"
            },
            "system_metrics": {
                "total_requests": system_state['total_requests'],
                "successful_responses": system_state['successful_responses'],
                "error_count": system_state['error_count'],
                "success_rate": system_state['successful_responses'] / max(system_state['total_requests'], 1),
                "appointments_booked": system_state.get('appointments_booked', 0),
                "sos_triggered": system_state.get('sos_triggered', 0),
                "llama_responses": system_state.get('llama_responses', 0),
                "fallback_responses": system_state.get('fallback_responses', 0)
            }
        }

        # Database health check
        try:
            with app.app_context():
                total_users = User.query.count()
                total_conversations = ConversationTurn.query.count()
                health_status["database"] = {
                    "status": "connected",
                    "total_users": total_users,
                    "total_conversations": total_conversations
                }
        except Exception as e:
            health_status["database"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # Overall health assessment
        critical_components = ['nlu_processor', 'response_generator', 'conversation_memory', 'database']
        if not all(system_status.get(comp, False) for comp in critical_components):
            health_status["status"] = "degraded"

        return jsonify(health_status)

    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route("/v1/admin/metrics", methods=["GET"])
@admin_required
def get_system_metrics():
    """Provides system-wide metrics for the admin dashboard."""
    try:
        # Get last 30 days of metrics
        thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
        recent_metrics = SystemMetrics.query.filter(SystemMetrics.metrics_date >= thirty_days_ago).order_by(SystemMetrics.metrics_date.desc()).all()

        metrics_data = [
            {
                "date": metric.metrics_date.isoformat(),
                "activeUsers": metric.total_active_users,
                "newUsers": metric.new_users_registered,
                "conversations": metric.total_conversations,
                "appointments": metric.appointments_booked,
                "orders": metric.orders_placed
            } for metric in recent_metrics
        ]

        # Get current high-level stats
        current_stats = {
            "totalUsers": User.query.count(),
            "totalDoctors": Doctor.query.count(),
            "totalPharmacies": Pharmacy.query.count(),
            "pendingGrievances": GrievanceReport.query.filter_by(status='Pending').count()
        }

        return jsonify({
            "success": True,
            "historicalMetrics": metrics_data,
            "currentStats": current_stats
        })

    except Exception as e:
        logger.error(f"❌ Metrics retrieval error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve system metrics"}), 500

@app.route("/v1/save-models", methods=["POST"])
def save_models_endpoint():
    """Manually trigger comprehensive model saving"""
    try:
        # Authentication check (basic - enhance in production)
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('ADMIN_API_KEY', 'admin_key_123'):
            return jsonify({"error": "Unauthorized"}), 401

        success = save_all_models()

        if success:
            return jsonify({
                "success": True,
                "message": "All AI models saved successfully",
                "timestamp": datetime.now().isoformat(),
                "models_saved": {
                    "nlu_processor": system_status['nlu_processor'],
                    "conversation_memory": system_status['conversation_memory'],
                    # Crisis detector model saving status removed
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "Some models failed to save - check logs for details",
                "timestamp": datetime.now().isoformat()
            }), 500

    except Exception as e:
        logger.error(f"❌ Model saving endpoint error: {e}")
        return jsonify({
            "success": False,
            "message": f"Error saving models: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/v1/system/status", methods=["GET"])
def system_status_endpoint():
    """Get detailed system status information"""
    return jsonify({
        "system_info": {
            "application_name": "Sehat Sahara Health Assistant",
            "version": "2.1.0",
            "startup_time": system_state['startup_time'].isoformat(),
            "current_time": datetime.now().isoformat(),
            "uptime_hours": round((datetime.now() - system_state['startup_time']).total_seconds() / 3600, 2)
        },
        "components": system_status,
        "api_ollama_integration": {
            "available": sehat_sahara_client.is_available,
            "model": sehat_sahara_client.client.model if sehat_sahara_client.is_available else "N/A",
            "base_url": sehat_sahara_client.client.base_url if sehat_sahara_client.is_available else "N/A"
        },
        "features": {
            "progressive_conversation_stages": True,
            "method_effectiveness_tracking": True,
            # Optimized crisis detection removed
            "professional_counselor_integration": False, # Removed as it's not relevant for Sehat Sahara
            "comprehensive_user_analytics": True,
            "persistent_conversation_memory": True,
            "enhanced_emotional_analysis": False, # Removed as it's not the focus
            "real_time_system_monitoring": True,
            "ollama_ai_enhancement": sehat_sahara_client.is_available,
            "task_oriented_actions": True, # New feature for Sehat Sahara
            "emergency_handling": True, # New feature for Sehat Sahara
            "sehat_sahara_api": sehat_sahara_client.is_available  # API availability status
        },
        "performance": {
            "total_requests": system_state['total_requests'],
            "successful_responses": system_state['successful_responses'],
            "error_count": system_state['error_count'],
            "success_rate": system_state['successful_responses'] / max(system_state['total_requests'], 1),
            "appointments_booked": system_state.get('appointments_booked', 0),
            "sos_triggered": system_state.get('sos_triggered', 0),
            "llama_responses": system_state.get('llama_responses', 0),
            "fallback_responses": system_state.get('fallback_responses', 0)
        }
    })

# Ollama-specific endpoints
# In chatbot.py (add this new route function)


@app.route("/v1/ollama/status", methods=["GET"])
def ollama_status():
    """Get Ollama integration status"""
    return jsonify({
        "ollama_available": sehat_sahara_client.is_available,
        "client_info": {
            "base_url": sehat_sahara_client.client.base_url,
            "model": sehat_sahara_client.client.model
        },
        "integration_working": system_status.get('ollama_llama3', False),
        "responses_generated": system_state.get('llama_responses', 0),
        "fallback_responses": system_state.get('fallback_responses', 0)
    })

@app.route("/v1/ollama/test", methods=["POST"])
def test_ollama():
    try:
        data = request.get_json() or {}
        test_message = data.get("message", "Book a doctor for tomorrow morning in my village").strip()
        if sehat_sahara_client and sehat_sahara_client.is_available:
            result = sehat_sahara_client.generate_action_json(test_message)
            if result:
                return jsonify({"success": True, "result": result})
            return jsonify({"success": False, "message": "No result"}), 500
        return jsonify({"success": False, "message": "API not available"}), 503
    except Exception as e:
        logger.error(f"Ollama test error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/v1/user/<user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    """Get user profile and health app statistics"""
    try:
        if hasattr(initialize_ai_components, '_conversation_memory'):
            user_summary = initialize_ai_components._conversation_memory.get_user_summary(user_id)
        else:
            user_summary = None

        if not user_summary:
            return jsonify({
                "error": "User not found",
                "status": "error"
            }), 404

        return jsonify({
            "profile": user_summary,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to retrieve user profile",
            "status": "error"
        }), 500
    
# Find the trigger_sos function and replace it entirely with this updated version:
@app.route("/v1/sos-trigger", methods=["POST"])
def trigger_sos():
    data = request.get_json() or {}
    user_id = data.get("userId")
    location = data.get("location")
    message = data.get("message")
    
    user = User.query.filter_by(patient_id=user_id).first()
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404
        
    event_id = str(uuid.uuid4())
    sos_events[event_id] = {
        "id": len(sos_events) + 1,
        "event_id": event_id,
        "patient_name": user.full_name,
        "patient_id": user.patient_id,
        "location": location,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "emergency_type": "medical_emergency"
    }
    
    logger.info(f"SOS Triggered by {user.full_name} ({user_id})")

    # --- START: NEW PUSH NOTIFICATION LOGIC ---
    try:
        # In a real app, you would filter for users with the 'saathi' role
        # For this demo, we notify all subscribed users.
        saathi_users = User.query.filter_by(role='saathi').all()
        if not saathi_users:
            logger.warning("SOS triggered, but no users with the 'saathi' role were found to notify.")
            return jsonify({"success": True, "message": "SOS event created, but no Saathis found."})

        saathi_user_ids = [user.id for user in saathi_users]


        subscriptions = PushSubscription.query.filter(PushSubscription.user_id.in_(saathi_user_ids)).all()
        if not subscriptions:
            logger.warning(f"Found {len(saathi_users)} Saathis, but none have active push subscriptions.")
        
        
        push_payload = json.dumps({
            "title": "🚨 EMERGENCY ALERT 🚨",
            "options": {
                "body": f"SOS from {user.full_name}. Needs immediate assistance!",
                "icon": "https://i.ibb.co/YDNmS1D/siren.png", # A different icon for SOS
                "data": {
                    "type": "sos", # IMPORTANT: We add a type to identify this notification
                    "url": "/saathi.html#notificationsScreen"
                },
                "tag": f"sos-alert-{event_id}"
            }
        })

        for sub in subscriptions:
            try:
                webpush(
                    subscription_info=json.loads(sub.subscription_info),
                    data=push_payload,
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims={"sub": "mailto:" + VAPID_CLAIM_EMAIL}
                )
                logger.info(f"Sent SOS push notification for event {event_id} to a Saathi device.")
            except WebPushException as ex:
                logger.error(f"Failed to send SOS push notification: {ex}")
                # If a subscription is expired, delete it from the DB
                if ex.response and ex.response.status_code in [404, 410]:
                    db.session.delete(sub)
                    db.session.commit()
                    logger.info("Deleted an expired push subscription.")
    except Exception as e:
        logger.error(f"An error occurred during SOS push notification dispatch: {e}")
    # --- END: NEW PUSH NOTIFICATION LOGIC ---
    
    return jsonify({"success": True, "message": "SOS event created.", "event_id": event_id})

@app.route("/v1/sos-events", methods=["GET"])
def get_sos_events():
    # Return all events, sorted with pending first
    pending_events = [e for e in sos_events.values() if e['status'] == 'pending']
    other_events = [e for e in sos_events.values() if e['status'] != 'pending']
    all_sorted = sorted(pending_events, key=lambda x: x['timestamp'], reverse=True) + sorted(other_events, key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({"success": True, "sos_events": all_sorted})

@app.route("/v1/sos-respond", methods=["POST"])
def respond_to_sos():
    data = request.get_json() or {}
    event_id = data.get("event_id")
    saathi_id = data.get("saathi_user_id") # You'd need a Saathi model for this
    
    if event_id in sos_events:
        sos_events[event_id]['status'] = 'responded'
        logger.info(f"Saathi {saathi_id} responded to SOS event {event_id}")
        return jsonify({"success": True, "message": "Response logged."})
    
    return jsonify({"success": False, "error": "Event not found"}), 404

@app.route('/v1/user/<user_id>/preferences', methods=['POST'])
def update_user_preferences(user_id):
    """Update user preferences"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No preferences data provided",
                "status": "error"
            }), 400

        # Update preferences in conversation memory
        if hasattr(initialize_ai_components, '_conversation_memory'):
            initialize_ai_components._conversation_memory.update_user_preferences(user_id, data)

        return jsonify({
            "message": "Preferences updated successfully",
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to update preferences",
            "status": "error"
        }), 500

@app.route('/v1/stats', methods=['GET'])
def get_system_stats():
    """Get Sehat Sahara system statistics"""
    try:
        # Get conversation memory stats
        if hasattr(initialize_ai_components, '_conversation_memory'):
            memory_stats = initialize_ai_components._conversation_memory.get_system_stats()
        else:
            memory_stats = {'total_users': 0, 'active_users_week': 0, 'total_conversations': 0}

        # Get database stats
        db_stats = {
            "total_users": User.query.count(),
            "active_users": User.query.filter_by(is_active=True).count(),
            "total_conversations": ConversationTurn.query.count(),
            "conversations_today": ConversationTurn.query.filter(
                ConversationTurn.timestamp >= datetime.now().date()
            ).count(),
            "emergency_calls_today": ConversationTurn.query.filter(
                ConversationTurn.urgency_level == 'emergency',
                ConversationTurn.timestamp >= datetime.now().date()
            ).count()
        }

        return jsonify({
            "system_stats": {
                "memory": memory_stats,
                "database": db_stats,
                "api_status": "available" if sehat_sahara_client.is_available else "unavailable",
                "service_name": "Sehat Sahara Health Assistant",
                "version": "2.1.0"
            },
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to retrieve system statistics",
            "status": "error"
        }), 500

@app.route("/v1/dashboard", methods=["GET"])
def get_dashboard():
    """
    Implemented dashboard endpoint with proper data structure matching frontend expectations
    """
    try:
        update_system_state('get_dashboard')

        current_user = get_current_user()
        if not current_user:
            user_id_param = request.args.get('userId')
            if user_id_param:
                current_user = User.query.filter_by(patient_id=user_id_param).first()
        
        if not current_user:
            return jsonify({"success": False, "message": "Authentication required"}), 401

        # Get next appointment
        next_appointment = Appointment.query.filter(
            Appointment.user_id == current_user.id,
            Appointment.status == 'scheduled',
            Appointment.appointment_datetime > datetime.now()
        ).order_by(Appointment.appointment_datetime.asc()).first()

        next_appointment_data = None
        if next_appointment:
            doctor = Doctor.query.get(next_appointment.doctor_id)
            next_appointment_data = {
                "doctorName": doctor.full_name if doctor else "Unknown Doctor",
                "dateTime": next_appointment.appointment_datetime.isoformat() + "Z"
            }

        # Get prescription count
        prescription_count = HealthRecord.query.filter_by(
            user_id=current_user.id,
            record_type='prescription'
        ).count()

        # Get recent health records
        health_records = HealthRecord.query.filter_by(user_id=current_user.id)\
            .order_by(HealthRecord.created_at.desc()).limit(5).all()

        health_records_data = []
        for record in health_records:
            record_data = {
                "date": record.created_at.isoformat(),
                "title": record.title,
                "description": record.description or "",
                "recordType": record.record_type,
                "imageData": record.image_data  # Add
            }
            if record.file_url:
                record_data["imageUrl"] = record.file_url
            health_records_data.append(record_data)

        return jsonify({
            "success": True,
            "fullName": current_user.full_name,
            "nextAppointment": next_appointment_data,
            "prescriptionCount": prescription_count,
            "healthRecords": health_records_data
        })

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        update_system_state('get_dashboard', success=False)
        return jsonify({"success": False, "message": "Failed to load dashboard"}), 500

@app.route("/v1/doctors", methods=["GET"])
def get_doctors():
    """Get list of doctors with optional specialty filter"""
    try:
        update_system_state('get_doctors')

        specialty = request.args.get('specialty', '').strip()

        query = Doctor.query.filter_by(is_active=True)

        if specialty:
            query = query.filter(Doctor.specialization.ilike(f'%{specialty}%'))

        doctors = query.all()

        doctors_data = []
        for doctor in doctors:
            doctors_data.append({
                # --- THIS IS THE FIX ---
                "id": doctor.id,  # Use the integer primary key
                "name": doctor.full_name,
                "specialization": doctor.specialization,
                "qualification": doctor.qualification,
                "experience": doctor.experience_years,
                "clinicName": doctor.clinic_name,
                "rating": doctor.average_rating,
                "languages": doctor.get_languages_spoken(),
                "doctor_id_str": doctor.doctor_id
            })

        return jsonify({
            "success": True,
            "doctors": doctors_data,
            "count": len(doctors_data)
        })

    except Exception as e:
        logger.error(f"Get doctors error: {e}")
        update_system_state('get_doctors', success=False)
        return jsonify({"success": False, "message": "Failed to retrieve doctors"}), 500

@app.route("/v1/appointments", methods=["GET"])
def get_appointments():
    """
    Fixed dateTime format to include 'Z' suffix for proper ISO 8601 UTC format
    """
    try:
        update_system_state('get_appointments')
        current_user = get_current_user()
        if not current_user:
            user_id_param = request.args.get('userId')
            if user_id_param:
                current_user = User.query.filter_by(patient_id=user_id_param).first()
        if not current_user:
            return jsonify({"success": False, "message": "Authentication required"}), 401
        
        appointments = Appointment.query.filter_by(user_id=current_user.id)\
            .order_by(Appointment.appointment_datetime.desc()).all()

        appointments_data = []
        for appt in appointments:
            doctor = Doctor.query.get(appt.doctor_id)
            appointments_data.append({
                "id": appt.appointment_id,
                "doctorName": doctor.full_name if doctor else "Unknown Doctor",
                "specialization": doctor.specialization if doctor else "",
                "dateTime": appt.appointment_datetime.isoformat() + "Z",
                "type": appt.appointment_type,
                "status": appt.status,
                "chiefComplaint": appt.chief_complaint
            })

        return jsonify({
            "success": True,
            "appointments": appointments_data
        })
    except Exception as e:
        logger.error(f"Get appointments error: {e}")
        return jsonify({"success": False, "message": "Failed to retrieve appointments"}), 500

@app.route("/v1/appointments/cancel", methods=["POST"])
def cancel_appointment():
    """Cancel an appointment"""
    try:
        update_system_state('cancel_appointment')

        current_user = get_current_user()
        if not current_user:
            return jsonify({"success": False, "message": "Authentication required"}), 401

        data = request.get_json() or {}
        appointment_id = data.get("appointmentId", "").strip()

        if not appointment_id:
            return jsonify({"success": False, "message": "Appointment ID is required"}), 400

        appointment = Appointment.query.filter_by(
            appointment_id=appointment_id,
            user_id=current_user.id
        ).first()

        if not appointment:
            return jsonify({"success": False, "message": "Appointment not found"}), 404

        if appointment.cancel_appointment("Cancelled by user"):
            db.session.commit()
            return jsonify({
                "success": True,
                "message": "Appointment cancelled successfully"
            })
        else:
            return jsonify({"success": False, "message": "Appointment cannot be cancelled"}), 400

    except Exception as e:
        logger.error(f"Cancel appointment error: {e}")
        update_system_state('cancel_appointment', success=False)
        return jsonify({"success": False, "message": "Failed to cancel appointment"}), 500

@app.route("/v1/booking-data", methods=["GET"])
def get_booking_data():
    """Get booking data from database in website-compatible format"""
    try:
        update_system_state('get_booking_data')

        # Get all active doctors from database
        doctors = Doctor.query.filter_by(is_active=True).all()

        # Group doctors by specialization using ACTUAL database names
        doctors_by_category = {}
        categories = set()

        for doctor in doctors:
            specialization = doctor.specialization or "General Physician"
            categories.add(specialization)

            if specialization not in doctors_by_category:
                doctors_by_category[specialization] = []

            # Create doctor object using ACTUAL database names
            doctor_data = {
                "id": doctor.id,
                "name": doctor.full_name,  # Use actual database name
                "languages": ", ".join(doctor.get_languages_spoken()) if doctor.get_languages_spoken() else "Hindi, English",
                "img": doctor.profile_image_url or f"https://placehold.co/100x100/{hash(doctor.full_name) % 16777215:06x}/FFFFFF?text={doctor.full_name[:2].upper()}",
                "specialization": specialization,
                "doctor_id": doctor.doctor_id,
                "qualification": doctor.qualification,
                "experience": doctor.experience_years,
                "clinic_name": doctor.clinic_name,
                "consultation_fee": doctor.consultation_fee or 200.0
            }

            doctors_by_category[specialization].append(doctor_data)

        # Convert categories set to sorted list
        categories_list = sorted(list(categories))

        # Create response in same format as website's appData.booking
        booking_data = {
            "categories": categories_list,
            "doctors": doctors_by_category,
            "slots": ["10:00 AM", "11:30 AM", "02:00 PM", "04:30 PM"],
            "modes": ["Video Call", "Audio Call", "Photo-based"]
        }

        return jsonify({
            "success": True,
            "booking_data": booking_data,
            "total_doctors": len(doctors),
            "total_categories": len(categories_list)
        })

    except Exception as e:
        logger.error(f"Get booking data error: {e}")
        update_system_state('get_booking_data', success=False)
        return jsonify({"success": False, "message": "Failed to retrieve booking data"}), 500

@app.route("/v1/pharmacies", methods=["GET"])
def get_pharmacies():
    """Get list of active pharmacies."""
    try:
        pharmacies = Pharmacy.query.filter_by(is_active=True).all()
        pharmacies_data = [{
            "id": p.id, "pharmacy_id": p.pharmacy_id, "name": p.name, "address": p.address,
            "phone": p.phone_number, "estimated_delivery_time": p.estimated_delivery_time,
            "services": {
                "homeDelivery": p.home_delivery, "onlinePayment": p.online_payment,
                "emergencyService": p.emergency_service
            },
            "rating": p.average_rating
        } for p in pharmacies]
        return jsonify({"success": True, "pharmacies": pharmacies_data, "count": len(pharmacies_data)})
    except Exception as e:
        logger.error(f"Get pharmacies error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to retrieve pharmacies"}), 500

@app.route("/v1/medicines", methods=["GET"])
def search_medicines():
    """Search for medicines"""
    try:
        update_system_state('search_medicines')

        search_term = request.args.get('search', '').strip()

        if not search_term:
            return jsonify({"success": False, "message": "Search term is required"}), 400

        # For now, return mock medicine data since we don't have a medicines table
        # In production, this would query a Medicine table
        mock_medicines = [
            {"name": "Paracetamol", "type": "Pain Relief", "dosage": "500mg", "price": 25.0},
            {"name": "Ibuprofen", "type": "Anti-inflammatory", "dosage": "400mg", "price": 35.0},
            {"name": "Amoxicillin", "type": "Antibiotic", "dosage": "500mg", "price": 85.0},
            {"name": "Cetirizine", "type": "Antihistamine", "dosage": "10mg", "price": 15.0},
            {"name": "Omeprazole", "type": "Antacid", "dosage": "20mg", "price": 45.0}
        ]

        # Filter medicines based on search term
        filtered_medicines = [
            med for med in mock_medicines
            if search_term.lower() in med["name"].lower()
        ]

        return jsonify({
            "success": True,
            "medicines": filtered_medicines,
            "count": len(filtered_medicines)
        })

    except Exception as e:
        logger.error(f"Search medicines error: {e}")
        update_system_state('search_medicines', success=False)
        return jsonify({"success": False, "message": "Failed to search medicines"}), 500

@app.route("/v1/orders", methods=["POST"])
def place_order():
    """Place a medicine order with robust authentication and validation."""
    try:
        data = request.get_json() or {}
        
        # --- FIX #1: ADD FALLBACK AUTHENTICATION ---
        # This makes it consistent with other endpoints.
        current_user = get_current_user()
        if not current_user:
            user_id_param = data.get("userId")
            if user_id_param:
                current_user = User.query.filter_by(patient_id=user_id_param).first()
        # --- END OF FIX ---

        if not current_user or getattr(current_user, 'role', 'patient') != 'patient':
            return jsonify({"success": False, "message": "Authentication required. Please log in as a patient."}), 401

        pharmacy_id_str = str(data.get("pharmacyId", "")).strip()
        items = data.get("items", [])
        delivery_address = data.get("deliveryAddress", "").strip()

        if not pharmacy_id_str or not items:
            return jsonify({"success": False, "message": "Pharmacy ID and items are required"}), 400

        pharmacy = Pharmacy.query.filter_by(id=pharmacy_id_str, is_active=True).first()
        if not pharmacy:
            return jsonify({"success": False, "message": "Pharmacy not found or is inactive"}), 404

        total_amount = sum(float(item.get("price", 0)) * int(item.get("quantity", 1)) for item in items)
        delivery_fee = 50.0 if pharmacy.home_delivery else 0.0

        order = MedicineOrder(
            user_id=current_user.id, pharmacy_id=pharmacy.id, items=json.dumps(items),
            total_amount=total_amount, delivery_fee=delivery_fee,
            delivery_address=delivery_address or current_user.get_full_address(),
            contact_phone=current_user.phone_number,
            payment_method=data.get("paymentMethod", "cod"), status="placed"
        )

        db.session.add(order)
        db.session.commit()

        logger.info(f"✅ Order {order.order_id} placed by user {current_user.patient_id} at {pharmacy.name}")
        return jsonify({
            "success": True, "message": "Order placed successfully", "orderId": order.order_id,
            "estimatedDelivery": pharmacy.estimated_delivery_time or "30-45 mins",
            "totalAmount": total_amount, "deliveryFee": delivery_fee
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Place order error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Failed to place order due to a server error."}), 500
    
@app.route("/v1/pharmacy/dashboard", methods=["GET"])
def get_pharmacy_dashboard():
    """Pharmacy dashboard endpoint to get orders and inventory alerts."""
    try:
        pharmacy = get_current_user()
        if not pharmacy or not isinstance(pharmacy, Pharmacy):
            return jsonify({"error": "Not authenticated as a pharmacy"}), 401

        new_orders = MedicineOrder.query.filter_by(pharmacy_id=pharmacy.id, status='placed').order_by(MedicineOrder.created_at.desc()).all()
        orders_data = []
        for order in new_orders:
            patient = User.query.get(order.user_id)
            orders_data.append({
                "id": order.order_id,
                "customer": patient.full_name if patient else "Unknown",
                "status": order.status,
                "totalAmount": order.total_amount,
                "createdAt": order.created_at.isoformat(),
                "items": order.get_items()
            })

        return jsonify({
            "success": True, "pharmacyName": pharmacy.name, "pharmacyId": pharmacy.pharmacy_id,
            "newOrdersCount": len(orders_data),
            "pendingDeliveriesCount": MedicineOrder.query.filter_by(pharmacy_id=pharmacy.id, status='out_for_delivery').count(),
            "orders": orders_data
        })
    except Exception as e:
        logger.error(f"Error fetching pharmacy dashboard: {e}", exc_info=True)
        return jsonify({"error": "Failed to load pharmacy dashboard"}), 500

@app.route("/v1/pharmacy/profile", methods=["GET"])
def get_pharmacy_profile():
    """Get pharmacy profile information"""
    try:
        pharmacy = get_current_user()
        if not pharmacy or not isinstance(pharmacy, Pharmacy):
            return jsonify({"error": "Not authenticated as a pharmacy"}), 401

        return jsonify({
            "success": True,
            "pharmacy": {
                "id": pharmacy.id, "pharmacyId": pharmacy.pharmacy_id, "name": pharmacy.name,
                "email": pharmacy.email, "phoneNumber": pharmacy.phone_number, "address": pharmacy.address,
                "estimated_delivery_time": pharmacy.estimated_delivery_time,
                "services": {
                    "homeDelivery": pharmacy.home_delivery, "onlinePayment": pharmacy.online_payment,
                    "emergencyService": pharmacy.emergency_service
                },
                "rating": pharmacy.average_rating
            }
        })
    except Exception as e:
        logger.error(f"Error fetching pharmacy profile: {e}", exc_info=True)
        return jsonify({"error": "Failed to load pharmacy profile"}), 500

@app.route("/v1/upload-prescription", methods=["POST"])
def upload_prescription():
    """Upload offline prescription with image and AI analysis"""
    try:
        update_system_state('upload_prescription')

        data = request.get_json() or {}
        user_id_str = data.get("userId", "").strip()
        provider_name = data.get("providerName", "").strip()
        image_data = data.get("imageData", "").strip()

        if not all([user_id_str, provider_name, image_data]):
            return jsonify({"error": "userId, providerName, and imageData are required"}), 400

        user = User.query.filter_by(patient_id=user_id_str, is_active=True).first()
        if not user:
            return jsonify({"error": "User not found"}), 401

        # Use AI to analyze the prescription image
        extracted_data = None
        if groq_scout and groq_scout.is_available:
            # Strip the data URL prefix if present
            if image_data.startswith("data:image/jpeg;base64,"):
                image_data_clean = image_data.split(",", 1)[1]
            else:
                image_data_clean = image_data
            extracted_data = groq_scout.interpret_prescription_image(image_data_clean, language="en")
            logger.info(f"AI extracted prescription data: {extracted_data}")

        # Prepare prescription data
        doctor_name = extracted_data.get('doctor_name', provider_name) if extracted_data else provider_name
        medications = extracted_data.get('medications', []) if extracted_data else []
        tests = extracted_data.get('tests', []) if extracted_data else []
        diagnosis = extracted_data.get('diagnosis', '') if extracted_data else ''

        # Store Base64 image data directly in database for cloud deployment compatibility
        try:
            # Clean the Base64 data (remove data URL prefix if present)
            image_data_clean = image_data.split(",", 1)[1] if "," in image_data else image_data

            # Validate Base64 format
            base64.b64decode(image_data_clean)
        except Exception as decode_error:
            logger.error(f"Failed to decode Base64 image data: {decode_error}")
            return jsonify({"error": "Invalid image data format"}), 400

        # Create HealthRecord for the prescription with Base64 data in database
        record = HealthRecord(
            user_id=user.id,
            record_type='prescription',
            title=f'Prescription from {doctor_name}',
            description=f'Uploaded prescription. Diagnosis: {diagnosis}' if diagnosis else 'Uploaded prescription image',
            file_type='image/jpeg',
            file_url=None,  # Set to None for cloud deployment
            image_data=image_data_clean,  # Store Base64 data in database
            test_date=datetime.now().date()
        )

        db.session.add(record)
        try:
            db.session.commit()
            logger.info(f"✅ Prescription uploaded for user {user.patient_id}, record_id: {record.record_id}")

            # Auto-generate medicine reminders from prescription
            prescription_data = {
                'prescription_id': record.record_id,
                'doctor_name': doctor_name,
                'medications': extracted_data.get('medications', []) if extracted_data else [],
                'diagnosis': diagnosis,
                'instructions': extracted_data.get('instructions', '') if extracted_data else '',
                'source': 'patient_upload',  # Mark as patient-uploaded prescription
                'upload_method': 'camera_scan',
                'ai_extracted': extracted_data is not None
            }

            # Add to conversation memory for reminder generation
            if hasattr(initialize_ai_components, '_conversation_memory'):
                initialize_ai_components._conversation_memory.add_prescription_summary(user.patient_id, prescription_data)

            logger.info(f"✅ Auto-generated medicine reminders for user {user.patient_id}")

        except Exception as commit_error:
            logger.error(f"❌ Failed to commit prescription upload for user {user.patient_id}: {commit_error}")
            db.session.rollback()
            return jsonify({"error": "Failed to save prescription to database"}), 500


    
        # --- NEW: Generate the summary text right here ---
        summary_response = response_generator.generate_prescription_summary_response(
            prescription_data, user.preferred_language
        )

        return jsonify({
            "success": True,
            "message": "Prescription uploaded and analyzed successfully",
            "recordId": record.record_id,
            "extractedData": {
                "doctor_name": doctor_name,
                "medications": medications,
                "tests": tests,
                "diagnosis": diagnosis
            },
            "formatted_response": summary_response  # <-- ADD THIS LINE
        })

    except Exception as e:
        logger.error(f"❌ Upload prescription error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('upload_prescription', success=False)
        db.session.rollback()
        return jsonify({"error": "Failed to upload prescription"}), 500

@app.route("/v1/admin/users", methods=["GET"])
@admin_required
def get_all_users():
    """Admin endpoint to get all user types except patients."""
    try:
        doctors = Doctor.query.all()
        pharmacies = Pharmacy.query.all()

        all_users = []
        for d in doctors:
            all_users.append({"id": d.id, "name": d.full_name, "role": "Doctor", "contact": d.email, "status": "Verified" if d.is_verified else "Unverified"})
        for p in pharmacies:
            all_users.append({"id": p.id, "name": p.name, "role": "Pharmacy", "contact": p.phone_number, "status": "Verified" if p.is_verified else "Unverified"})

        return jsonify({"success": True, "users": all_users})
    except Exception as e:
        logger.error(f"Error fetching all users: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve users"}), 500


@app.route("/v1/admin/grievances", methods=["GET"])
@admin_required
def get_grievances():
    """Admin endpoint to get all grievance reports with patient names using a join."""
    try:
        # One query: join GrievanceReport -> User for patient name
        results = db.session.query(GrievanceReport, User.full_name)\
            .join(User, GrievanceReport.user_id == User.id)\
            .order_by(GrievanceReport.created_at.desc())\
            .all()

        grievances_data = []
        for report, patient_name in results:
            grievances_data.append({
                "id": report.report_id,
                "patient": patient_name or "Unknown",
                "subject": report.reason,
                "priority": report.priority,
                "status": report.status,
                "date": report.created_at.isoformat()
            })
        return jsonify({"success": True, "grievances": grievances_data})
    except Exception as e:
        logger.error(f"Error fetching grievances: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve grievances"}), 500


@app.route("/v1/doctor/dashboard", methods=["GET"])
def get_doctor_dashboard():
    """Doctor dashboard endpoint to get their specific appointments."""
    try:
        # Get doctor from session
        doctor_id_str = session.get('doctor_id')
        if not doctor_id_str:
            return jsonify({"error": "Not authenticated as doctor"}), 401

        doctor = Doctor.query.filter_by(doctor_id=doctor_id_str).first()

        
        if not doctor:
            return jsonify({"error": "Doctor not found"}), 404

        

        appointments = Appointment.query.filter(
            Appointment.doctor_id == doctor.id
        ).order_by(Appointment.appointment_datetime.asc()).all()

        appointments_data = []
        for appt in appointments:
            patient = User.query.get(appt.user_id)
            appointments_data.append({
                "id": appt.appointment_id,
                "patient": patient.full_name if patient else "Unknown",
                "patientId": patient.patient_id if patient else None,
                "time": appt.appointment_datetime.strftime('%I:%M %p'),
                "dateTime": appt.appointment_datetime.isoformat() + "Z", # <-- ADD THIS LINE
                "type": appt.appointment_type,
                "status": appt.status
            })

        return jsonify({
            "success": True,
            "doctorName": doctor.full_name,
            "appointments": appointments_data
        })
    except Exception as e:
        logger.error(f"Error fetching doctor dashboard: {e}", exc_info=True)
        return jsonify({"error": "Failed to load doctor dashboard"}), 500

@app.route("/v1/doctor/profile", methods=["GET"])
def get_doctor_profile():
    """Get doctor profile information"""
    try:
        # In chatbot1.py, inside the get_doctor_profile function, around line 1948
        doctor_id_str = session.get('doctor_id')
        if not doctor_id_str:
            return jsonify({"error": "Not authenticated as doctor"}), 401

        doctor = Doctor.query.filter_by(doctor_id=doctor_id_str).first()
       
        if not doctor:
            return jsonify({"error": "Doctor not found"}), 404

        return jsonify({
            "success": True,
            "doctor": {
                "id": doctor.id,
                "doctorId": doctor.doctor_id,
                "fullName": doctor.full_name,
                "email": doctor.email,
                "specialization": doctor.specialization,
                "qualification": doctor.qualification,
                "experience": doctor.experience_years,
                "clinicName": doctor.clinic_name,
                "phoneNumber": doctor.phone_number,
                "profileImageUrl": doctor.profile_image_url
            }
        })
    except Exception as e:
        logger.error(f"Error fetching doctor profile: {e}", exc_info=True)
        return jsonify({"error": "Failed to load doctor profile"}), 500


@app.route("/v1/agora/token", methods=["POST"])
def get_agora_token():
    """Generates an Agora RTC token for secure channel access."""
    # --- START: ADD DETAILED LOGGING ---
    logger.info("--- /v1/agora/token endpoint hit ---")
    logger.info(f"Request Headers: {dict(request.headers)}") # Log all incoming headers
    logger.info(f"Session BEFORE get_current_user: {dict(session)}") # Log session content
    # --- END: ADD DETAILED LOGGING ---
    try:
        # --- Basic Security Check ---
        user = get_current_user() # Use your existing helper
        # --- ADD LOGGING AFTER get_current_user ---
        logger.info(f"get_current_user returned: {user}")
        # --- END LOGGING ---
        if not user:
             logger.warning("Unauthorized attempt to get Agora token - get_current_user returned None.") # More specific log
             return jsonify({"success": False, "error": "Authentication required"}), 401

        data = request.get_json() or {}
        channel_name = data.get("channelName")
        # Use 0 for auto-assigned UID by Agora, common for simple cases.
        # Or, you could assign specific UIDs based on user.id if needed later.
        uid = data.get("uid", 0)

        # Basic validation
        if not channel_name:
            logger.warning("Agora token request missing channelName.")
            return jsonify({"success": False, "error": "channelName is required"}), 400

        if not AGORA_APP_ID or not AGORA_APP_CERTIFICATE:
             logger.error("Agora App ID or Certificate not configured on the backend.")
             return jsonify({"success": False, "error": "Agora service not configured properly on the server."}), 503

        # Set role (Publisher allows sending/receiving streams)
        role = 1 
        # Calculate expiration timestamp
        current_timestamp = int(time.time())
        privilege_expired_ts = current_timestamp + TOKEN_EXPIRATION_SEC

        # --- Generate the Token ---
        token = RtcTokenBuilder.buildTokenWithUid(
            AGORA_APP_ID,
            AGORA_APP_CERTIFICATE,
            channel_name,
            uid, # Use 0 for auto-assigned UID
            role,
            privilege_expired_ts
        )

        logger.info(f"Generated Agora RTC token for user {user.id} (UID: {uid}) joining channel: {channel_name}")
        return jsonify({"success": True, "token": token})

    except Exception as e:
        logger.error(f"❌ Error generating Agora token: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Failed to generate Agora token due to a server error."}), 500


# Scheduled tasks and cleanup
def cleanup_on_exit():
    """Cleanup tasks on application shutdown"""
    logger.info("🔄 Application shutdown - performing cleanup...")
    try:
        with app.app_context():
            # Save all models
            save_all_models()

            # Update final metrics
            track_system_metrics()

            # Close database connections
            db.session.close()

        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)

# Periodic tasks (every 24 hours)
def run_periodic_tasks():
    """Run periodic maintenance tasks"""
    try:
        logger.info("🔄 Running periodic maintenance tasks...")

        # Track daily metrics
        track_system_metrics()

        # Clean up old data (keep 90 days)
        if hasattr(initialize_ai_components, '_conversation_memory'):
            initialize_ai_components._conversation_memory.cleanup_old_data(days_to_keep=90)

        # Save models
        save_all_models()

        logger.info("✅ Periodic maintenance completed")
    except Exception as e:
        logger.error(f"❌ Error in periodic tasks: {e}")

def schedule_periodic_tasks():
    """Schedule periodic tasks to run every 24 hours"""
    def task_scheduler():
        while True:
            time.sleep(24 * 60 * 60)  # 24 hours
            run_periodic_tasks()

    scheduler_thread = threading.Thread(target=task_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("✅ Periodic task scheduler started")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested resource does not exist",
        "available_endpoints": [
            "/v1/health", "/v1/register", "/v1/login", "/v1/logout",
            "/v1/predict", "/v1/book-doctor", "/v1/complete-appointment",
            "/v1/history", "/v1/user-stats", "/v1/ollama/status", "/v1/ollama/test",
            "/v1/dashboard", "/v1/doctors", "/v1/appointments", "/v1/appointments/cancel",
            "/v1/pharmacies", "/v1/medicines", "/v1/orders", "/v1/upload-prescription",
            "/v1/prescription-summary", "/v1/medicine-reminders", "/v1/enhanced-sos",
            "/v1/button-action", "/v1/user-progress", "/v1/post-appointment-feedback",
            "/v1/admin/users", "/v1/admin/grievances",
            "/v1/doctor/dashboard", "/v1/pharmacy/dashboard",
            "/v1/test-prescription", "/v1/test-openrouter-nlu"  # Test endpoints
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The requested HTTP method is not supported for this endpoint"
    }), 405

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Request too large",
        "message": "The request payload is too large"
    }), 413

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later."
    }), 429

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later.",
        "support_resources": {
            "emergency_services": "108",
            "health_helpline": "104"
        }
    }), 500
# Serve static HTML files
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    if filename.endswith('.html'):
        return send_from_directory('.', filename)
    return send_from_directory('.', filename)
# Enhanced startup display with Ollama information
def display_startup_info():
    print("=" * 100)
    print("SEHAT SAHARA HEALTH ASSISTANT - Task-Oriented App Navigator")
    print("=" * 100)
    print()
    print("FEATURES:")
    print(" * Action JSON responses for app navigation")
    print(" * Emergency handling with TRIGGER_SOS (108)")
    print(" * Appointment booking, health records, pharmacy guidance")
    print()
    print("API ENDPOINTS:")
    print(" * POST /v1/register")
    print(" * POST /v1/login")
    print(" * POST /v1/logout")
    print(" * POST /v1/predict  (returns {response, action, parameters})")
    print(" * POST /v1/book-doctor")
    print(" * POST /v1/history")
    print(" * POST /v1/user-stats")
    print(" * GET  /v1/health")
    print(" * GET  /v1/ollama/status")
    print(" * POST /v1/ollama/test")
    print(" * GET  /v1/dashboard")
    print(" * GET  /v1/doctors")
    print(" * GET  /v1/appointments")
    print(" * POST /v1/appointments/cancel")
    print(" * GET  /v1/pharmacies")
    print(" * GET  /v1/medicines")
    print(" * POST /v1/orders")
    print(" * GET  /v1/admin/users")
    print(" * GET  /v1/admin/grievances")
    print(" * GET  /v1/doctor/dashboard")
    print(" * GET  /v1/pharmacy/dashboard")
    print(" * GET  /v1/test-prescription")
    print(" * GET|POST /v1/test-openrouter-nlu  (test new AI features)")
    print()
    print("=" * 100)
    print("SYSTEM READY - SEHAT SAHARA ACTIVE")
    print("=" * 100)

if __name__ == "__main__":
    # Display startup information
    display_startup_info()

    # Initialize periodic tasks
    schedule_periodic_tasks()

    # Track initial system startup
    with app.app_context():
        try:
            track_system_metrics()
        except Exception as e:
            logger.error(f"Failed to track startup metrics: {e}")

    # Start the Flask application

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)



























