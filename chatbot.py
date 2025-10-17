from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import os
import logging
import traceback
from datetime import datetime, timedelta
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
    ConversationTurn, UserSession, GrievanceReport, SystemMetrics, init_database, get_user_statistics
)

# Import enhanced AI components with Ollama integration
from nlu_processor import ProgressiveNLUProcessor
from ko import ProgressiveResponseGenerator
# Remove crisis detector import and usage
# from optimized_crisis_detector import OptimizedCrisisDetector  # removed

from api_ollama_integration import sehat_sahara_client, groq_scout


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

# Initialize Flask application with enhanced configuration
app = Flask(__name__)
# Enhanced CORS configuration for better compatibility
CORS(app, supports_credentials=True, resources={
    r"/*": {  # Covers ALL routes including /v1/*
        "origins": [
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "http://localhost:3000",
            "https://saharasaathi.netlify.app",
            "https://sahara-sathi.onrender.com",
            "https://sehat-sahara.onrender.com",
            "*"  # Allow all origins for static files
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Authorization"]
    }
})


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

# WebRTC signaling messages storage (in-memory for demo)
webrtc_messages = {}

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


        # Initialize NLU Processor
        logger.info("🧠 Initializing Progressive NLU Processor...")
        nlu_processor = ProgressiveNLUProcessor(model_path=nlu_model_path)
        system_status['nlu_processor'] = True
        logger.info("✅ NLU Processor initialized successfully")

        # Initialize Response Generator
        logger.info("💬 Initializing Progressive Response Generator...")
        response_generator = ProgressiveResponseGenerator()
        system_status['response_generator'] = True
        logger.info("✅ Response Generator initialized successfully")

        # Initialize Conversation Memory
        logger.info("Initializing Progressive Conversation Memory...")
        from conversation_memory import ProgressiveConversationMemory  # keep memory
        global conversation_memory
        conversation_memory = ProgressiveConversationMemory()
        system_status['conversation_memory'] = True
        logger.info("Conversation Memory initialized successfully")

        logger.info("✅ All AI components initialized for Sehat Sahara.")
        return True

    except Exception as e:
        logger.error(f"❌ Critical error initializing AI components: {e}")
        logger.error(traceback.format_exc())

        # Initialize minimal fallback components
        try:
            logger.info("🔄 Attempting to initialize fallback components...")
            nlu_processor = ProgressiveNLUProcessor()
            response_generator = ProgressiveResponseGenerator()
            from conversation_memory import ProgressiveConversationMemory
            conversation_memory = ProgressiveConversationMemory()
            logger.info("⚠️ Fallback components initialized (limited functionality)")
            return False
        except Exception as fallback_error:
            logger.error(f"❌ Failed to initialize even fallback components: {fallback_error}")
            nlu_processor = None
            response_generator = None
            conversation_memory = None
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

        if conversation_memory and system_status['conversation_memory']:
            conversation_memory.save_to_file(os.path.join(models_path, 'conversation_memory.json'))
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
        role = session.get('role')
        if not role:
            return None

        if role == 'doctor':
            doctor_id = session.get('doctor_id')
            if doctor_id:
                return Doctor.query.filter_by(doctor_id=doctor_id, is_active=True).first()

        elif role == 'pharmacy':
            pharmacy_id = session.get('pharmacy_id')
            if pharmacy_id:
                return Pharmacy.query.filter_by(pharmacy_id=pharmacy_id, is_active=True).first()

        else: # Default to patient
            user_id = session.get('user_id')
            if user_id:
                user = User.query.filter_by(id=user_id, is_active=True).first()
                if user:
                    user.role = getattr(user, 'role', 'patient')
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

@app.route("/v1/login", methods=["POST"])
def login():
    try:
        data = request.get_json() or {}
        login_identifier = data.get("patientId", "").strip()
        password = data.get("password", "")
        role = data.get("role", "patient").strip().lower()

        if not login_identifier or not password:
            return jsonify({"success": False, "message": "Identifier and password are required."}), 400

        user_obj = None
        user_type = 'patient'

        if role == 'doctor':
            user_obj = Doctor.query.filter(
                (Doctor.email == login_identifier.lower()) | (Doctor.doctor_id == login_identifier.upper())
            ).first()
            user_type = 'doctor'
        elif role == 'pharmacy':
            user_obj = Pharmacy.query.filter(
                (Pharmacy.email == login_identifier.lower()) | (Pharmacy.pharmacy_id == login_identifier.upper())
            ).first()
            user_type = 'pharmacy'
        else:
            user_obj = User.query.filter(
                (User.email == login_identifier.lower()) | (User.patient_id == login_identifier.upper())
            ).first()
            user_type = 'patient'

        if user_obj and user_obj.is_active and user_obj.check_password(password):
            session.permanent = True
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
                response_data['redirect'] = "/store.html" # Redirect to pharmacy dashboard
                user_data = {"pharmacyId": user_obj.pharmacy_id, "username": user_obj.name, "role": "pharmacy"}
            else: # Patient
                session['patient_id'] = user_obj.patient_id
                user_obj.update_last_login()
                db.session.commit()
                response_data['message'] = f"Welcome back, {user_obj.full_name}!"
                user_data = {"patientId": user_obj.patient_id, "username": user_obj.full_name, "role": "patient"}

            response_data['user'] = user_data
            create_user_session(user_obj, request.environ)
            logger.info(f"✅ {user_type.capitalize()} login successful: {login_identifier}")
            return jsonify(response_data)

        logger.warning(f"⚠️ Failed login attempt for identifier: '{login_identifier}' as {role}")
        return jsonify({"success": False, "message": "Invalid credentials or account inactive."}), 401

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
        progress_summary = conversation_memory.get_user_progress_summary(user_id)

        followup_needed = conversation_memory.check_post_appointment_followup(user_id)

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
        history_turns = conversation_memory.get_conversation_context(user_id, turns=4) if conversation_memory else []
        nlu_history = []
        for turn in history_turns:
            nlu_history.append({'role': 'user', 'content': turn.get('user_message', '')})
            try:
                bot_response_json = json.loads(turn.get('bot_response', '{}'))
                bot_content = bot_response_json.get('response', '')
            except (json.JSONDecodeError, AttributeError):
                bot_content = turn.get('bot_response', '')
            nlu_history.append({'role': 'assistant', 'content': bot_content})

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

        if conversation_memory:
            conversation_memory.complete_post_appointment_feedback(user_id)

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

@app.route("/v1/medicine-reminders", methods=["POST"])
def manage_medicine_reminders():
    """
    New comprehensive endpoint to manage medicine reminders (get, add, update_adherence)
    """
    try:
        # Critical safety check for AI components
        if not conversation_memory:
            logger.error("FATAL: Conversation memory component not initialized. Service unavailable.")
            return jsonify({"error": "The AI assistant is currently unavailable. Please try again later."}), 503

        data = request.get_json() or {}
        user_id = data.get("userId", "").strip()
        action = data.get("action", "get").strip().lower()  # 'get', 'add', 'update', 'delete'

        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        if action == "get":
            # Get user's medicine reminders
            reminders = conversation_memory.get_medicine_reminders(user_id)
            alerts = conversation_memory.get_reminder_alerts(user_id)

            return jsonify({
                "success": True,
                "reminders": reminders,
                "pending_alerts": alerts,
                "total_reminders": len(reminders)
            })

        elif action == "add":
            # Add new medicine reminder
            medicine_data = {
                'name': data.get('medicine_name'),
                'dosage': data.get('dosage'),
                'frequency': data.get('frequency'),
                'times': data.get('times', []),
                'duration_days': data.get('duration_days', 30),
                'start_date': data.get('start_date', datetime.now().strftime('%Y-%m-%d')),
                'instructions': data.get('instructions', '')
            }

            conversation_memory.schedule_medicine_reminder(user_id, medicine_data)

            return jsonify({
                "success": True,
                "message": "Medicine reminder scheduled successfully"
            })

        elif action == "update_adherence":
            # Update medicine adherence
            medicine_name = data.get('medicine_name')
            taken_time = data.get('taken_time', datetime.now().strftime('%H:%M'))

            conversation_memory.update_reminder_adherence(user_id, medicine_name, taken_time)

            return jsonify({
                "success": True,
                "message": f"Medicine {medicine_name} marked as taken"
            })

        else:
            return jsonify({"error": "Invalid action. Use 'get', 'add', or 'update_adherence'"}), 400

    except Exception as e:
        logger.error(f"Medicine reminders error: {e}")
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
        prescription_data = conversation_memory.get_prescription_summary(user_id, prescription_id)

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
    task = conversation_memory.get_current_task(user_id)
    # If a booking is already in progress, use its context
    if task.get('task') == 'appointment_booking' and task.get('context'):
        return task.get('context')

    # Otherwise, start a new booking task with an empty context
    new_context = {}
    conversation_memory.set_current_task(user_id, 'appointment_booking', new_context)
    return new_context

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
    task = conversation_memory.get_current_task(user_id)
    if task.get('task') == 'symptom_triage' and task.get('context'):
        return task.get('context')
    
    # Start a new task if one isn't active
    new_context = {
        'turn_count': 0,
        'symptoms_reported': [] # <-- New: A list to store user's answers
    }
    if initial_symptom:
        new_context['symptoms_reported'].append(initial_symptom)

    conversation_memory.set_current_task(user_id, 'symptom_triage', new_context)
    return new_context

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

        nlu_understanding = nlu_processor.understand_user_intent(user_message, sehat_sahara_mode=True)
        primary_intent = nlu_understanding.get('primary_intent', 'general_inquiry')
        
        task = conversation_memory.get_current_task(current_user.patient_id)
        task_in_progress = task.get('task')

        # --- FIX #1: FLEXIBLE STATE MANAGEMENT ---
        # If the user has a new, unrelated intent, cancel the old task.
        # This prevents the booking/symptom flow from "hijacking" the conversation.
        related_booking_intents = ['appointment_booking', 'SELECT_SPECIALTY', 'SELECT_DOCTOR', 'SELECT_DATE', 'SELECT_TIME', 'SELECT_MODE']
        is_unrelated_intent = (
            task_in_progress and
            primary_intent not in related_booking_intents and
            button_action not in related_booking_intents and
            primary_intent != task_in_progress
        )
        if is_unrelated_intent:
            logger.warning(f"User {current_user.patient_id} switched intent from '{task_in_progress}' to '{primary_intent}'. Completing old task.")
            conversation_memory.complete_task(current_user.patient_id)
            task_in_progress = None # Reset task_in_progress to allow new task to start
        # --- END OF FIX #1 ---

        ai_message_override = user_message

        # --- TASK MANAGEMENT LOGIC ---
        if task_in_progress == 'appointment_booking':
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
                logger.warning(f"Booking flow stalled. User input '{user_message}' did not match step '{last_step}'. Re-prompting.")
                ai_message_override = f"CONTEXT: User provided an unexpected response. Please re-ask them to select an option for the current step: '{last_step}'."
            
            conversation_memory.set_current_task(current_user.patient_id, 'appointment_booking', booking_context)
        
        # This logic is now correctly accessed if the booking task was cleared above
        
        elif task_in_progress == 'symptom_triage':
            symptom_context = _get_or_create_symptom_context(current_user.patient_id)
    # --- FIX #2: Append the new symptom to our memory ---
            symptom_context['symptoms_reported'].append(user_message)
            turn_count = symptom_context.get('turn_count', 0)
    
    # We'll ask 2 follow-up questions total
            if turn_count < 2:
                ai_message_override = f"CONTEXT: This is a symptom check conversation. User replied '{user_message}'. Acknowledge their answer and ask one more simple clarifying question based on the conversation history."
                symptom_context['turn_count'] = turn_count + 1
                conversation_memory.set_current_task(current_user.patient_id, 'symptom_triage', symptom_context)
            else:
        # --- FIX #2: Pass all collected symptoms to the final prompt ---
                reported_symptoms_str = '; '.join(symptom_context['symptoms_reported'])
                ai_message_override = (
                    f"CONTEXT: The user has reported the following symptoms: '{reported_symptoms_str}'. "
                    "You have asked enough questions. Provide a simple, safe home remedy for these symptoms. "
                    "Then, you MUST include this exact disclaimer: 'This is not medical advice. For a proper diagnosis, please consult a doctor.' "
                    "Your final action should be 'CONTINUE_CONVERSATION'."
                    )
                conversation_memory.complete_task(current_user.patient_id) 

        # This logic is also now correctly accessed
        else:
            if primary_intent == 'appointment_booking':
                booking_context = _get_or_create_booking_context(current_user.patient_id)
                booking_context.clear()
                booking_context['last_step'] = 'ask_specialty'
                available_specialties = _get_available_specialties_from_db()
                specialty_buttons = ', '.join([f'{{"text": "{s}", "action": "SELECT_SPECIALTY", "parameters": {{"specialty": "{s}"}}}}' for s in available_specialties])
                ai_message_override = f"CONTEXT: User wants to book an appointment. Ask them to select a specialty using these buttons: [{specialty_buttons}]"
                conversation_memory.set_current_task(current_user.patient_id, 'appointment_booking', booking_context)

            
            elif primary_intent == 'symptom_triage':
    # --- FIX #2: Pass the initial symptom when creating the context ---
                 symptom_context = _get_or_create_symptom_context(current_user.patient_id, initial_symptom=user_message)
                 symptom_context['turn_count'] = 1
                 ai_message_override = f"CONTEXT: Start of a symptom check. User said '{user_message}'. Acknowledge their symptom and ask your first clarifying question (e.g., 'For how long?' or 'Is it a sharp or dull pain?')."
                 conversation_memory.set_current_task(current_user.patient_id, 'symptom_triage', symptom_context)
        
        # ... rest of the function remains the same ...
        # --- AI RESPONSE GENERATION ---
        history = conversation_memory.get_conversation_context(current_user.patient_id, turns=8)
        context = {"user_intent": primary_intent, "context_history": history}
        
        action_payload = None
        if sehat_sahara_client and sehat_sahara_client.is_available:
            action_payload_str = sehat_sahara_client.generate_sehatsahara_response(
                user_message=ai_message_override, context=context
            )
            if action_payload_str:
                try:
                    action_payload = json.loads(action_payload_str)
                    if action_payload.get("action") == "FINALIZE_BOOKING":
                        booking_context = _get_or_create_booking_context(current_user.patient_id)
                        doc_id = booking_context.get('doctor_id')
                        appt_date = booking_context.get('date')
                        appt_time_str = booking_context.get('time')
                        doctor = Doctor.query.filter_by(doctor_id=doc_id).first()
                        if all([doctor, appt_date, appt_time_str]):
                            appt_datetime = datetime.fromisoformat(f"{appt_date}T{appt_time_str}:00")
                            new_appt = Appointment(user_id=current_user.id, doctor_id=doctor.id, appointment_datetime=appt_datetime, appointment_type=booking_context.get('mode', 'Video Call'), chief_complaint=f"Consultation for {booking_context.get('specialty')}", status='confirmed')
                            db.session.add(new_appt)
                            db.session.commit()
                            update_system_state('book_doctor', appointments_booked=1)
                            action_payload['response'] = f"Your appointment with Dr. {doctor.full_name} is confirmed for {appt_datetime.strftime('%A, %b %d at %I:%M %p')}."
                            action_payload['action'] = 'BOOKING_SUCCESS'
                            conversation_memory.complete_task(current_user.patient_id)
                        else:
                            action_payload['response'] = "I'm sorry, there was an error with the booking details. Let's start over."
                            action_payload['action'] = 'BOOKING_FAILED'
                            conversation_memory.complete_task(current_user.patient_id)
                except json.JSONDecodeError:
                    action_payload = None
        
        if not action_payload:
            logger.warning(f"AI failed or unavailable. Using local fallback for intent: {primary_intent}")
            action_payload = response_generator.generate_response(user_message=user_message, nlu_result=nlu_understanding, user_context={'user_id': current_user.patient_id}, conversation_history=history)

        # --- FINAL PROCESSING & SAVING ---
        action_payload_str = json.dumps(action_payload, ensure_ascii=False)
        turn_record = ConversationTurn(user_id=current_user.id, user_message=user_message, bot_response=action_payload_str, detected_intent=primary_intent, action_triggered=action_payload.get('action'))
        db.session.add(turn_record)
        db.session.commit()
        
        response_time = time.time() - start_time
        logger.info(f"User {current_user.patient_id} processed in {response_time:.2f}s")
        
        return jsonify(action_payload)

    except Exception as e:
        logger.error(f"FATAL ERROR in /predict endpoint: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({"response": "I'm having a technical issue. Please try again.", "action": "SHOW_APP_FEATURES", "interactive_buttons": []}), 500

def get_booking_data_internal():
    """Internal helper to get booking data from database"""
    try:
        # Get all active doctors from database
        doctors = Doctor.query.filter_by(is_active=True).all()

        # Group doctors by specialization using actual database names
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

        return {
            "success": True,
            "booking_data": booking_data,
            "total_doctors": len(doctors),
            "total_categories": len(categories_list)
        }

    except Exception as e:
        logger.error(f"Get booking data internal error: {e}")
        return None


@app.route("/v1/book-doctor", methods=["POST"])
def book_doctor():
    try:
        update_system_state('book_doctor')
        data = request.get_json() or {}

        # --- CORRECTED LOGIC ---
        # First, try to get the user from the active session
        user = get_current_user()

        # If no user is found in the session, fall back to the userId from the request
        if not user:
            user_id_param = data.get("userId")
            if user_id_param:
                # Query the database using the patient_id
                user = User.query.filter_by(patient_id=user_id_param).first()

        # If still no user is found, then authentication fails
        if not user:
            return jsonify({"success": False, "message": "Authentication required or user not found"}), 401
        # --- END OF CORRECTION ---

        doctor_id_str = (data.get("doctorId") or "").strip()
        appointment_dt = (data.get("appointmentDatetime") or "").strip()
        appointment_type = (data.get("appointmentType") or "consultation").strip()
        chief_complaint = (data.get("chiefComplaint") or "").strip()
        symptoms = data.get("symptoms") or []

        if not doctor_id_str or not appointment_dt:
            return jsonify({"success": False, "message": "doctorId and appointmentDatetime are required"}), 400

        doctor = Doctor.query.filter(
            (Doctor.doctor_id == doctor_id_str) | (Doctor.id == doctor_id_str)
        ).first()
        if not doctor:
            return jsonify({"success": False, "message": "Doctor not found"}), 404

        # Parse datetime in ISO format
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

        # Update session counters
        try:
            session_record_id = session.get('session_record_id')
            if session_record_id:
                s = UserSession.query.get(session_record_id)
                if s:
                    s.appointments_booked_in_session += 1
        except Exception:
            pass

        db.session.commit()
        update_system_state('book_doctor', appointments_booked=1)

        return jsonify({
            "success": True,
            "message": "Appointment created.",
            "appointment": {
                "appointmentId": appt.appointment_id,
                "doctor": {"id": doctor.doctor_id, "name": doctor.full_name, "specialization": doctor.specialization},
                "datetime": appt.appointment_datetime.isoformat() + "Z", # Append 'Z' for UTC
                # --- END OF CORRECTION ---
                "type": appt.appointment_type,
                "status": appt.status
            }
        })

    except Exception as e:
        logger.error(f"Book doctor error: {e}")
        logger.error(traceback.format_exc())
        update_system_state('book_doctor', success=False)
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
        
        if conversation_memory:
            conversation_memory.update_appointment_status(
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
        user_summary = conversation_memory.get_user_summary(user_id_str) if conversation_memory else {}

        response_data = {
            "success": True,
            "history": chat_log,
            "summary": {
                "total_conversations": current_user.total_conversations,
                "current_stage": conversation_memory.get_conversation_stage_db(current_user.patient_id),
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
        user_summary = conversation_memory.get_user_summary(user_id)
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
        conversation_memory.update_user_preferences(user_id, data)

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
        memory_stats = conversation_memory.get_system_stats()

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
                "recordType": record.record_type
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
                "id": doctor.doctor_id,
                "name": doctor.full_name,
                "specialization": doctor.specialization,
                "qualification": doctor.qualification,
                "experience": doctor.experience_years,
                "clinicName": doctor.clinic_name,
                "rating": doctor.average_rating,
                "languages": doctor.get_languages_spoken()
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
        current_user = get_current_user()

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
            conversation_memory.add_prescription_summary(user.patient_id, prescription_data)

            logger.info(f"✅ Auto-generated medicine reminders for user {user.patient_id}")

        except Exception as commit_error:
            logger.error(f"❌ Failed to commit prescription upload for user {user.patient_id}: {commit_error}")
            db.session.rollback()
            return jsonify({"error": "Failed to save prescription to database"}), 500


    
        return jsonify({
            "success": True,
            "message": "Prescription uploaded and analyzed successfully",
            "recordId": record.record_id,
            "extractedData": {
                "doctor_name": doctor_name,
                "medications": medications,
                "tests": tests,
                "diagnosis": diagnosis
            }
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

        today_start = datetime.now().date()
        today_end = today_start + timedelta(days=1)

        appointments = Appointment.query.filter(
            Appointment.doctor_id == doctor.id,
            Appointment.appointment_datetime >= today_start,
            Appointment.appointment_datetime < today_end
        ).order_by(Appointment.appointment_datetime.asc()).all()

        appointments_data = []
        for appt in appointments:
            patient = User.query.get(appt.user_id)
            appointments_data.append({
                "id": appt.appointment_id,
                "patient": patient.full_name if patient else "Unknown",
                "time": appt.appointment_datetime.strftime('%I:%M %p'),
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


    

# WebRTC signaling endpoints
@app.route("/v1/webrtc/<message_type>", methods=["POST"])
def webrtc_signaling(message_type):
    """Handle WebRTC signaling messages"""
    try:
        data = request.get_json() or {}
        appointment_id = data.get("appointmentId")
        message_data = data.get("data")

        if not appointment_id or message_data is None:
            return jsonify({"error": "appointmentId and data required"}), 400

        if appointment_id not in webrtc_messages:
            webrtc_messages[appointment_id] = []

        webrtc_messages[appointment_id].append({
            "type": message_type,
            "data": message_data,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"WebRTC message stored: {message_type} for appointment {appointment_id}")
        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"WebRTC signaling error: {e}")
        return jsonify({"error": "Failed to process signaling message"}), 500

@app.route("/v1/webrtc/poll", methods=["GET"])
def webrtc_poll():
    """Poll for WebRTC signaling messages"""
    try:
        appointment_id = request.args.get("appointmentId")
        if not appointment_id:
            return jsonify([])

        messages = webrtc_messages.get(appointment_id, [])
        # Return messages and clear them after sending
        webrtc_messages[appointment_id] = []
        return jsonify(messages)

    except Exception as e:
        logger.error(f"WebRTC poll error: {e}")
        return jsonify({"error": "Failed to poll messages"}), 500


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
        if conversation_memory:
            conversation_memory.cleanup_old_data(days_to_keep=90)

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
            "/v1/test-prescription"  # Test endpoint
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

