"""
Sehat Sahara Health Assistant Database Models
Database schema for health app features: appointments, doctors, health records, pharmacies
"""

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import uuid

db = SQLAlchemy()

class User(db.Model):
    """User model for Sehat Sahara patients"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    # MODIFIED: Increased length to accommodate full names
    patient_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone_number = db.Column(db.String(15), nullable=True)
    # Change it to this to enforce uniqueness and make it required:
    # To this (remove unique=True):
    full_name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='patient')
    
    # Location information for rural patients
    village = db.Column(db.String(100), nullable=True)
    district = db.Column(db.String(100), nullable=True)
    state = db.Column(db.String(100), default='Punjab')
    pincode = db.Column(db.String(10), nullable=True)
    
    # Account management
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=False)
    phone_verified = db.Column(db.Boolean, default=False)
    
    # App usage tracking
    total_conversations = db.Column(db.Integer, default=0)
    preferred_language = db.Column(db.String(10), default='hi')  # hi, pa, en
    app_version = db.Column(db.String(20), nullable=True)
    
    # Emergency contact
    emergency_contact_name = db.Column(db.String(100), nullable=True)
    emergency_contact_phone = db.Column(db.String(15), nullable=True)
    
    # Enhanced fields
    timezone = db.Column(db.String(50), default='Asia/Kolkata')
    notification_preferences = db.Column(db.Text)  # JSON object for notification settings

    # Conversation state tracking
    current_conversation_stage = db.Column(db.String(100), default='general')  # Track current conversation stage
    
    # Relationships
    conversation_turns = db.relationship('ConversationTurn', backref='user', lazy=True, cascade='all, delete-orphan')
    appointments = db.relationship('Appointment', backref='user', lazy=True, cascade='all, delete-orphan')
    health_records = db.relationship('HealthRecord', backref='user', lazy=True, cascade='all, delete-orphan')
    user_sessions = db.relationship('UserSession', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_notification_preferences(self):
        """Get notification preferences as dict"""
        if self.notification_preferences:
            try:
                return json.loads(self.notification_preferences)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_notification_preferences(self, prefs_dict):
        """Set notification preferences from dict"""
        self.notification_preferences = json.dumps(prefs_dict)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.now()

    def update_conversation_stage(self, stage: str):
        """Update current conversation stage"""
        self.current_conversation_stage = stage
    
    def get_full_address(self):
        """Get formatted full address"""
        address_parts = [self.village, self.district, self.state, self.pincode]
        return ', '.join(filter(None, address_parts))
    
    def __repr__(self):
        return f'<User {self.patient_id}>'

class Doctor(db.Model):
    """Doctor model for healthcare providers"""
    __tablename__ = 'doctors'

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.String(15), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(100), nullable=False)

    # --- ADD THIS LINE ---
    password_hash = db.Column(db.String(256), nullable=True) # Making it nullable for now

    specialization = db.Column(db.String(100), nullable=False)
    qualification = db.Column(db.String(200), nullable=True)
    experience_years = db.Column(db.Integer, default=0)

    # Contact information
    phone_number = db.Column(db.String(15), nullable=True)
    email = db.Column(db.String(120), nullable=True, unique=True)

    # --- AND ADD THIS LINE ---
    profile_image_url = db.Column(db.String(500), nullable=True)

    # Practice information
    clinic_name = db.Column(db.String(200), nullable=True)
    clinic_address = db.Column(db.Text, nullable=True)
    consultation_fee = db.Column(db.Float, default=0.0)

    # Availability (JSON format)
    availability_schedule = db.Column(db.Text)  # JSON object with weekly schedule

    # Status and ratings
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    average_rating = db.Column(db.Float, default=0.0)
    total_ratings = db.Column(db.Integer, default=0)

    # Languages spoken
    languages_spoken = db.Column(db.Text)  # JSON array

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    appointments = db.relationship('Appointment', backref='doctor', lazy=True)

    # --- ADD THESE METHODS FOR PASSWORD HANDLING ---
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_availability_schedule(self):
        """Get availability schedule as dict"""
        if self.availability_schedule:
            try:
                return json.loads(self.availability_schedule)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_availability_schedule(self, schedule_dict):
        """Set availability schedule from dict"""
        self.availability_schedule = json.dumps(schedule_dict)
    
    def get_languages_spoken(self):
        """Get languages spoken as list"""
        if self.languages_spoken:
            try:
                return json.loads(self.languages_spoken)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_languages_spoken(self, languages_list):
        """Set languages spoken from list"""
        self.languages_spoken = json.dumps(languages_list)
    
    def is_available_at(self, datetime_obj):
        """Check if doctor is available at given datetime"""
        schedule = self.get_availability_schedule()
        day_name = datetime_obj.strftime('%A').lower()
        
        if day_name not in schedule:
            return False
        
        day_schedule = schedule[day_name]
        if not day_schedule.get('available', False):
            return False
        
        time_str = datetime_obj.strftime('%H:%M')
        start_time = day_schedule.get('start_time', '09:00')
        end_time = day_schedule.get('end_time', '17:00')
        
        return start_time <= time_str <= end_time
    
    def __repr__(self):
        return f'<Doctor {self.full_name} - {self.specialization}>'

class Appointment(db.Model):
    """Appointment model for patient-doctor bookings"""
    __tablename__ = 'appointments'
    
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False, index=True)
    
    # Appointment details
    appointment_datetime = db.Column(db.DateTime, nullable=False, index=True)
    duration_minutes = db.Column(db.Integer, default=30)
    appointment_type = db.Column(db.String(50), default='consultation')  # consultation, follow_up, emergency
    
    # Status tracking
    status = db.Column(db.String(50), default='scheduled', index=True)  # scheduled, confirmed, completed, cancelled, no_show
    booking_source = db.Column(db.String(50), default='app')  # app, phone, walk_in
    
    # Patient information
    chief_complaint = db.Column(db.Text, nullable=True)  # Main reason for visit
    symptoms = db.Column(db.Text, nullable=True)  # JSON array of symptoms
    
    # Consultation details
    consultation_notes = db.Column(db.Text, nullable=True)
    prescription = db.Column(db.Text, nullable=True)  # JSON object
    follow_up_required = db.Column(db.Boolean, default=False)
    follow_up_date = db.Column(db.DateTime, nullable=True)
    
    # Payment information
    consultation_fee = db.Column(db.Float, default=0.0)
    payment_status = db.Column(db.String(50), default='pending')  # pending, paid, failed
    payment_method = db.Column(db.String(50), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    cancelled_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Cancellation details
    cancellation_reason = db.Column(db.String(200), nullable=True)
    cancelled_by = db.Column(db.String(50), nullable=True)  # patient, doctor, system
    
    def get_symptoms(self):
        """Get symptoms as list"""
        if self.symptoms:
            try:
                return json.loads(self.symptoms)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_symptoms(self, symptoms_list):
        """Set symptoms from list"""
        self.symptoms = json.dumps(symptoms_list)
    
    def get_prescription(self):
        """Get prescription as dict"""
        if self.prescription:
            try:
                return json.loads(self.prescription)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_prescription(self, prescription_dict):
        """Set prescription from dict"""
        self.prescription = json.dumps(prescription_dict)
    
    def can_be_cancelled(self):
        """Check if appointment can be cancelled"""
        if self.status in ['completed', 'cancelled']:
            return False
        
        # Can't cancel if appointment is within 2 hours
        time_until_appointment = self.appointment_datetime - datetime.now()
        return time_until_appointment.total_seconds() > 7200  # 2 hours
    
    def cancel_appointment(self, reason, cancelled_by='patient'):
        """Cancel the appointment"""
        if self.can_be_cancelled():
            self.status = 'cancelled'
            self.cancellation_reason = reason
            self.cancelled_by = cancelled_by
            self.cancelled_at = datetime.now()
            return True
        return False
    
    def complete_appointment(self, notes=None, prescription=None):
        """Mark appointment as completed"""
        self.status = 'completed'
        self.completed_at = datetime.now()
        if notes:
            self.consultation_notes = notes
        if prescription:
            self.set_prescription(prescription)
    
    def __repr__(self):
        return f'<Appointment {self.appointment_id}: {self.status}>'

class HealthRecord(db.Model):
    """Health record model for storing patient medical documents"""
    __tablename__ = 'health_records'
    
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    
    # Foreign key
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Record details
    record_type = db.Column(db.String(50), nullable=False, index=True)  # lab_report, prescription, x_ray, etc.
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # File information
    file_url = db.Column(db.String(500), nullable=True)
    file_name = db.Column(db.String(200), nullable=True)
    file_size = db.Column(db.Integer, nullable=True)  # in bytes
    file_type = db.Column(db.String(50), nullable=True)  # pdf, jpg, png, etc.
    image_data = db.Column(db.Text, nullable=True)  # base64 encoded image data
    
    # Medical information
    test_date = db.Column(db.DateTime, nullable=True)
    doctor_name = db.Column(db.String(100), nullable=True)
    hospital_name = db.Column(db.String(200), nullable=True)
    
    # Record metadata
    is_critical = db.Column(db.Boolean, default=False)
    is_shared = db.Column(db.Boolean, default=False)
    tags = db.Column(db.Text, nullable=True)  # JSON array of tags
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def get_tags(self):
        """Get tags as list"""
        if self.tags:
            try:
                return json.loads(self.tags)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_tags(self, tags_list):
        """Set tags from list"""
        self.tags = json.dumps(tags_list)
    
    def get_file_size_formatted(self):
        """Get formatted file size"""
        if not self.file_size:
            return "Unknown"
        
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        else:
            return f"{self.file_size / (1024 * 1024):.1f} MB"
    
    def __repr__(self):
        return f'<HealthRecord {self.title}: {self.record_type}>'

class Pharmacy(db.Model):
    """Pharmacy model for medicine shops and availability"""
    __tablename__ = 'pharmacies'

    id = db.Column(db.Integer, primary_key=True)
    pharmacy_id = db.Column(db.String(15), unique=True, nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)

    # ADD THIS LINE FOR LOGIN
    password_hash = db.Column(db.String(256), nullable=True)

    # Contact information
    phone_number = db.Column(db.String(15), nullable=True)
    email = db.Column(db.String(120), nullable=True, unique=True)

    # Location information
    address = db.Column(db.Text, nullable=False)
    village = db.Column(db.String(100), nullable=True)
    district = db.Column(db.String(100), nullable=True)
    state = db.Column(db.String(100), default='Punjab')
    pincode = db.Column(db.String(10), nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)

    # Business information
    license_number = db.Column(db.String(50), nullable=True)
    owner_name = db.Column(db.String(100), nullable=True)

    # Operating hours (JSON format)
    operating_hours = db.Column(db.Text)

    # Services offered
    home_delivery = db.Column(db.Boolean, default=False)
    online_payment = db.Column(db.Boolean, default=False)
    emergency_service = db.Column(db.Boolean, default=False)

    # Status and ratings
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    average_rating = db.Column(db.Float, default=0.0)
    total_ratings = db.Column(db.Integer, default=0)

    # Delivery information
    estimated_delivery_time = db.Column(db.String(100), nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def get_operating_hours(self):
        if self.operating_hours:
            try: return json.loads(self.operating_hours)
            except json.JSONDecodeError: return {}
        return {}

    def set_operating_hours(self, hours_dict):
        self.operating_hours = json.dumps(hours_dict)

    # --- ADD THESE METHODS FOR PASSWORD HANDLING ---
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        """Check password against hash"""
        if self.password_hash is None:
            return False
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<Pharmacy {self.name}>'



class MedicineOrder(db.Model):
    """Medicine order model for pharmacy orders"""
    __tablename__ = 'medicine_orders'

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)

    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    pharmacy_id = db.Column(db.Integer, db.ForeignKey('pharmacies.id'), nullable=False, index=True)

    # Order details
    items = db.Column(db.Text, nullable=False)  # JSON array of medicine items
    total_amount = db.Column(db.Float, default=0.0)
    delivery_fee = db.Column(db.Float, default=0.0)

    # Status tracking
    status = db.Column(db.String(50), default='placed', index=True)  # placed, preparing, out_for_delivery, delivered, cancelled
    payment_status = db.Column(db.String(50), default='pending')  # pending, paid, failed
    payment_method = db.Column(db.String(50), nullable=True)

    # Delivery information
    delivery_address = db.Column(db.Text, nullable=True)
    delivery_instructions = db.Column(db.Text, nullable=True)
    estimated_delivery_time = db.Column(db.String(100), nullable=True)  # e.g., "30-45 mins"

    # Contact information
    contact_phone = db.Column(db.String(15), nullable=True)
    alternative_contact = db.Column(db.String(15), nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    delivered_at = db.Column(db.DateTime, nullable=True)
    cancelled_at = db.Column(db.DateTime, nullable=True)

    # Cancellation details
    cancellation_reason = db.Column(db.String(200), nullable=True)
    cancelled_by = db.Column(db.String(50), nullable=True)  # customer, pharmacy, system

    def get_items(self):
        """Get order items as list"""
        if self.items:
            try:
                return json.loads(self.items)
            except json.JSONDecodeError:
                return []
        return []

    def set_items(self, items_list):
        """Set order items from list"""
        self.items = json.dumps(items_list)

    def can_be_cancelled(self):
        """Check if order can be cancelled"""
        if self.status in ['delivered', 'cancelled', 'out_for_delivery']:
            return False
        return True

    def cancel_order(self, reason, cancelled_by='customer'):
        """Cancel the order"""
        if self.can_be_cancelled():
            self.status = 'cancelled'
            self.cancellation_reason = reason
            self.cancelled_by = cancelled_by
            self.cancelled_at = datetime.now()
            return True
        return False

    def mark_delivered(self):
        """Mark order as delivered"""
        self.status = 'delivered'
        self.delivered_at = datetime.now()

    def __repr__(self):
        return f'<MedicineOrder {self.order_id}: {self.status}>'

class ConversationTurn(db.Model):
    """Simplified conversation turn for chat history"""
    __tablename__ = 'conversation_turns'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Conversation content
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now, index=True)
    
    # NLU Analysis results
    detected_intent = db.Column(db.String(50), index=True)
    intent_confidence = db.Column(db.Float, default=0.0)
    language_detected = db.Column(db.String(10), default='hi')
    
    # App action taken
    action_triggered = db.Column(db.String(100), nullable=True)
    action_parameters = db.Column(db.Text, nullable=True)  # JSON
    
    # Context and tracking
    urgency_level = db.Column(db.String(20), default='low')
    context_entities = db.Column(db.Text)  # JSON of extracted entities
    
    # Performance metrics
    response_time_ms = db.Column(db.Integer)
    user_satisfaction_rating = db.Column(db.Integer)  # 1-5 if provided by user
    
    # Enhanced fields
    turn_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    session_id = db.Column(db.String(36), nullable=True)
    
    def get_context_entities(self):
        """Get context entities as dict"""
        if self.context_entities:
            try:
                return json.loads(self.context_entities)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_context_entities(self, entities_dict):
        """Set context entities from dict"""
        self.context_entities = json.dumps(entities_dict)
    
    def get_action_parameters(self):
        """Get action parameters as dict"""
        if self.action_parameters:
            try:
                return json.loads(self.action_parameters)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_action_parameters(self, params_dict):
        """Set action parameters from dict"""
        self.action_parameters = json.dumps(params_dict)
    
    def __repr__(self):
        return f'<ConversationTurn {self.id}: {self.detected_intent}>'

class UserSession(db.Model):
    """Track user sessions for analytics"""
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Session tracking
    session_start = db.Column(db.DateTime, default=datetime.now)
    session_end = db.Column(db.DateTime)
    session_duration_minutes = db.Column(db.Float)
    
    # Session details
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    device_type = db.Column(db.String(50))  # mobile, tablet, desktop
    app_version = db.Column(db.String(20))
    
    # Activity metrics for this session
    conversations_in_session = db.Column(db.Integer, default=0)
    actions_triggered_in_session = db.Column(db.Integer, default=0)
    appointments_booked_in_session = db.Column(db.Integer, default=0)
    
    # Enhanced session tracking
    session_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    login_method = db.Column(db.String(20), default='password')
    last_activity = db.Column(db.DateTime, default=datetime.now)
    is_active = db.Column(db.Boolean, default=True)
    
    def end_session(self):
        """End the session and calculate duration"""
        self.session_end = datetime.now()
        self.is_active = False
        
        if self.session_start:
            duration = self.session_end - self.session_start
            self.session_duration_minutes = duration.total_seconds() / 60
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_expired(self, hours=24):
        """Check if session is expired"""
        if not self.last_activity:
            return True
        
        expiry_time = self.last_activity + timedelta(hours=hours)
        return datetime.now() > expiry_time
    
    def __repr__(self):
        return f'<UserSession {self.session_id}>'

class GrievanceReport(db.Model):
    """Model for tracking user-reported issues."""
    __tablename__ = 'grievance_reports'

    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    subject_user_id = db.Column(db.Integer) # ID of doctor/pharmacy being reported

    reason = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    priority = db.Column(db.String(50), default='Medium') # Low, Medium, High
    status = db.Column(db.String(50), default='Pending') # Pending, Resolved, Dismissed

    created_at = db.Column(db.DateTime, default=datetime.now)
    resolved_at = db.Column(db.DateTime, nullable=True)

    def resolve(self):
        self.status = 'Resolved'
        self.resolved_at = datetime.now()

    def __repr__(self):
        return f'<GrievanceReport {self.report_id}>'

class SystemMetrics(db.Model):
    """Model for daily system-wide analytics for the admin panel."""
    __tablename__ = 'system_metrics'

    id = db.Column(db.Integer, primary_key=True)
    metrics_date = db.Column(db.Date, unique=True, nullable=False)

    total_active_users = db.Column(db.Integer, default=0)
    new_users_registered = db.Column(db.Integer, default=0)
    total_conversations = db.Column(db.Integer, default=0)
    appointments_booked = db.Column(db.Integer, default=0)
    prescriptions_issued = db.Column(db.Integer, default=0)
    orders_placed = db.Column(db.Integer, default=0)
    grievances_filed = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f'<SystemMetrics for {self.metrics_date}>'

class SystemConfiguration(db.Model):
    """Store system configuration settings"""
    __tablename__ = 'system_configuration'
    
    id = db.Column(db.Integer, primary_key=True)
    config_key = db.Column(db.String(100), unique=True, nullable=False)
    config_value = db.Column(db.Text)
    config_type = db.Column(db.String(20), default='string')
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    @classmethod
    def get_config(cls, key, default=None):
        """Get configuration value"""
        config = cls.query.filter_by(config_key=key).first()
        if not config:
            return default
        
        if config.config_type == 'json':
            try:
                return json.loads(config.config_value)
            except json.JSONDecodeError:
                return default
        elif config.config_type == 'boolean':
            return config.config_value.lower() in ['true', '1', 'yes']
        elif config.config_type == 'integer':
            try:
                return int(config.config_value)
            except ValueError:
                return default
        elif config.config_type == 'float':
            try:
                return float(config.config_value)
            except ValueError:
                return default
        
        return config.config_value
    
    @classmethod
    def set_config(cls, key, value, config_type='string', description=None):
        """Set configuration value"""
        config = cls.query.filter_by(config_key=key).first()
        
        if config_type == 'json' and not isinstance(value, str):
            value = json.dumps(value)
        elif config_type in ['boolean', 'integer', 'float']:
            value = str(value)
        
        if config:
            config.config_value = value
            config.config_type = config_type
            config.updated_at = datetime.now()
            if description:
                config.description = description
        else:
            config = cls(
                config_key=key,
                config_value=value,
                config_type=config_type,
                description=description
            )
        
        db.session.add(config)
        db.session.commit()
        return config

# Database initialization function
def init_database(app):
    """Initialize database with app context"""
    with app.app_context():
        db.create_all()
        
        # Create default system configurations for Sehat Sahara
        default_configs = [
            ('emergency_numbers', json.dumps([
                {'name': 'Ambulance', 'number': '108', 'description': 'Emergency ambulance service'},
                {'name': 'Police', 'number': '100', 'description': 'Police emergency'},
                {'name': 'Fire', 'number': '101', 'description': 'Fire emergency'},
                {'name': 'Women Helpline', 'number': '1091', 'description': 'Women in distress'}
            ]), 'json', 'Emergency contact numbers'),
            ('max_session_duration_hours', '24', 'integer', 'Maximum session duration in hours'),
            ('supported_languages', json.dumps(['hi', 'pa', 'en']), 'json', 'Supported languages'),
            ('app_version', '1.0.0', 'string', 'Current app version'),
            ('maintenance_mode', 'false', 'boolean', 'System maintenance mode flag'),
            ('default_consultation_fee', '200', 'float', 'Default consultation fee in INR'),
            ('appointment_cancellation_hours', '2', 'integer', 'Hours before appointment when cancellation is not allowed')
        ]
        
        for key, value, config_type, description in default_configs:
            if not SystemConfiguration.query.filter_by(config_key=key).first():
                SystemConfiguration.set_config(key, value, config_type, description)

        # Seed initial doctors if they don't exist
        seed_initial_doctors()

        # --- ADD THESE TWO LINES ---
        # Seed initial pharmacies if they don't exist
        seed_initial_pharmacies()
        seed_initial_saathis() # <-- ADD THIS LINE

        # Set default time slots configuration if not exists
        if not SystemConfiguration.query.filter_by(config_key='available_time_slots').first():
            default_time_slots = [
                {"text": "10:00 AM", "hour": 10, "minute": 0},
                {"text": "11:30 AM", "hour": 11, "minute": 30},
                {"text": "02:00 PM", "hour": 14, "minute": 0},
                {"text": "04:30 PM", "hour": 16, "minute": 30}
            ]
            SystemConfiguration.set_config(
                'available_time_slots',
                default_time_slots,
                'json',
                'Available time slots for appointment booking'
            )

        print("✅ Sehat Sahara database tables created successfully")
        print("✅ Default system configurations initialized")
        print("✅ Initial doctors seeded successfully")

def get_user_statistics(user_id: int) -> dict:
    """Get comprehensive user statistics for Sehat Sahara"""
    user = User.query.get(user_id)
    if not user:
        return {}
    
    # Get appointment statistics
    total_appointments = Appointment.query.filter_by(user_id=user_id).count()
    completed_appointments = Appointment.query.filter_by(user_id=user_id, status='completed').count()
    upcoming_appointments = Appointment.query.filter(
        Appointment.user_id == user_id,
        Appointment.status == 'scheduled',
        Appointment.appointment_datetime > datetime.now()
    ).count()
    
    # Get health records count
    health_records_count = HealthRecord.query.filter_by(user_id=user_id).count()
    
    # Get conversation statistics
    total_conversations = ConversationTurn.query.filter_by(user_id=user_id).count()
    
    # Get session statistics
    total_sessions = UserSession.query.filter_by(user_id=user_id).count()
    avg_session_duration = db.session.query(db.func.avg(UserSession.session_duration_minutes))\
        .filter_by(user_id=user_id).scalar() or 0
    
    return {
        'user_info': {
            'patient_id': user.patient_id,
            'full_name': user.full_name,
            'preferred_language': user.preferred_language,
            'location': user.get_full_address(),
            'member_since': user.created_at,
            'last_active': user.last_login,
            'total_sessions': total_sessions,
            'avg_session_duration': round(avg_session_duration, 2)
        },
        'appointments': {
            'total_appointments': total_appointments,
            'completed_appointments': completed_appointments,
            'upcoming_appointments': upcoming_appointments,
            'completion_rate': (completed_appointments / total_appointments * 100) if total_appointments > 0 else 0
        },
        'health_records': {
            'total_records': health_records_count
        },
        'app_usage': {
            'total_conversations': total_conversations,
            'app_version': user.app_version
        }
    }

def cleanup_expired_sessions():
    """Clean up expired user sessions"""
    expired_cutoff = datetime.now() - timedelta(hours=24)
    
    expired_sessions = UserSession.query.filter(
        UserSession.last_activity < expired_cutoff,
        UserSession.is_active == True
    ).all()
    
    for session in expired_sessions:
        session.end_session()
    
    db.session.commit()
    return len(expired_sessions)

def get_system_health():
    """Get overall system health metrics for Sehat Sahara"""
    try:
        # Basic counts
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        total_doctors = Doctor.query.filter_by(is_active=True).count()
        total_pharmacies = Pharmacy.query.filter_by(is_active=True).count()

        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_conversations = ConversationTurn.query.filter(
            ConversationTurn.timestamp >= recent_cutoff
        ).count()

        recent_appointments = Appointment.query.filter(
            Appointment.created_at >= recent_cutoff
        ).count()

        # Active sessions
        active_sessions = UserSession.query.filter_by(is_active=True).count()

        return {
            'database_healthy': True,
            'total_users': total_users,
            'active_users': active_users,
            'total_doctors': total_doctors,
            'total_pharmacies': total_pharmacies,
            'recent_conversations': recent_conversations,
            'recent_appointments': recent_appointments,
            'active_sessions': active_sessions,
            'last_checked': datetime.now().isoformat()
        }

    except Exception as e:
        return {
            'database_healthy': False,
            'error': str(e),
            'last_checked': datetime.now().isoformat()
        }

def seed_initial_doctors():
    """Seed initial doctors into the database"""
    from werkzeug.security import generate_password_hash

    doctors_to_add = [
        {
            "doctor_id": "DOC001",
            "full_name": "Aarav Sharma",
            "specialization": "General Physician",
            "email": "aarav.sharma@clinic.com",
            "password": "password123",
            "profile_image_url": "https://images.unsplash.com/photo-1612349317150-e413f6a5b16e?q=80&w=2070&auto=format&fit=crop"
        },
        {
            "doctor_id": "DOC002",
            "full_name": "Priya Singh",
            "specialization": "Dermatologist",
            "email": "priya.singh@clinic.com",
            "password": "password123",
            "profile_image_url": "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?q=80&w=2070&auto=format&fit=crop"
        },
        {
            "doctor_id": "DOC003",
            "full_name": "Rohan Mehta",
            "specialization": "Child Specialist",
            "email": "rohan.mehta@clinic.com",
            "password": "password123",
            "profile_image_url": "https://images.unsplash.com/photo-1537368910025-700350796527?q=80&w=2070&auto=format&fit=crop"
        }
    ]

    for doc_data in doctors_to_add:
        existing_doctor = Doctor.query.filter_by(doctor_id=doc_data["doctor_id"]).first()
        hashed_password = generate_password_hash(doc_data["password"])
        if not existing_doctor:
            new_doctor = Doctor(
                doctor_id=doc_data["doctor_id"],
                full_name=doc_data["full_name"],
                specialization=doc_data["specialization"],
                email=doc_data["email"],
                password_hash=hashed_password,
                profile_image_url=doc_data["profile_image_url"],
                is_active=True
            )
            db.session.add(new_doctor)
        else:
            # Update fields if they don't match
            updated = False
            if existing_doctor.specialization != doc_data["specialization"]:
                existing_doctor.specialization = doc_data["specialization"]
                updated = True
            if existing_doctor.password_hash != hashed_password.decode('utf-8'):
                existing_doctor.password_hash = hashed_password.decode('utf-8')
                updated = True
            if updated:
                db.session.add(existing_doctor)

    db.session.commit()

def seed_initial_saathis():
    """Seed initial Saathi users into the database."""
    saathis_to_add = [
        {
            "patient_id": "SAATHI001",
            "full_name": "Rohan Sharma (Saathi)",
            "email": "saathi.rohan@sehara.com",
            "password": "password123",
            "role": "saathi" # CRITICAL: Set the role correctly
        }
        # You can add more Saathi users here
    ]

    for saathi_data in saathis_to_add:
        existing_user = User.query.filter_by(patient_id=saathi_data["patient_id"]).first()
        if not existing_user:
            new_saathi = User(
                patient_id=saathi_data["patient_id"],
                full_name=saathi_data["full_name"],
                email=saathi_data["email"],
                role=saathi_data["role"],
                is_active=True
            )
            new_saathi.set_password(saathi_data["password"])
            db.session.add(new_saathi)
    
    db.session.commit()

def seed_initial_pharmacies():
    """Seed initial pharmacies into the database"""
    pharmacies_to_add = [
        {
            "pharmacy_id": "PHAR001", "name": "Apollo Pharmacy", "email": "apollo.pharmacy@pharm.com",
            "password": "password123", "phone_number": "+91 98765 43210",
            "address": "123 Health St, Wellness Colony, Kolkata, WB 700091", "estimated_delivery_time": "30-45 mins"
        },
        {
            "pharmacy_id": "PHAR002", "name": "Frank Ross", "email": "frankross.pharmacy@pharm.com",
            "password": "password123", "phone_number": "+91 98765 43211",
            "address": "456 Medical Ave, Health District, Mumbai, MH 400001", "estimated_delivery_time": "45-60 mins"
        },
        {
            "pharmacy_id": "PHAR003", "name": "Wellness Forever", "email": "wellnessforever.pharmacy@pharm.com",
            "password": "password123", "phone_number": "+91 98765 43212",
            "address": "789 Care Blvd, Wellness Park, Delhi, DL 110001", "estimated_delivery_time": "20-35 mins"
        }
    ]

    for pharm_data in pharmacies_to_add:
        existing_pharmacy = Pharmacy.query.filter_by(pharmacy_id=pharm_data["pharmacy_id"]).first()
        if not existing_pharmacy:
            new_pharmacy = Pharmacy(
                pharmacy_id=pharm_data["pharmacy_id"], name=pharm_data["name"], email=pharm_data["email"],
                phone_number=pharm_data["phone_number"], address=pharm_data["address"],
                estimated_delivery_time=pharm_data.get("estimated_delivery_time"), is_active=True
            )
            new_pharmacy.set_password(pharm_data["password"])
            db.session.add(new_pharmacy)

    db.session.commit()
# In enhanced_database_models.py, add this new class

class PushSubscription(db.Model):
    """Stores web push subscription details for users"""
    __tablename__ = 'push_subscriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    subscription_info = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    user = db.relationship('User', backref=db.backref('push_subscriptions', lazy=True, cascade='all, delete-orphan'))

    def __repr__(self):
        return f'<PushSubscription for User {self.user_id}>'
# In enhanced_database_models.py

# --- ADD THIS NEW CLASS AT THE END OF THE FILE ---
class KeyValueStore(db.Model):
    """A simple key-value store to save the memory JSON to the database."""
    __tablename__ = 'key_value_store'
    key = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.Text, nullable=False)
