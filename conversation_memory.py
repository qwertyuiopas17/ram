"""
Sehat Sahara Health Assistant Conversation Memory
Simplified conversation tracking and user context management for health app navigation
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class UserProfile:
    """Enhanced user profile for Sehat Sahara Health Assistant with progress tracking"""
    user_id: str
    patient_id: str = ""
    full_name: str = ""
    preferred_language: str = "hi"  # hi, pa, en
    location: str = ""

    # Conversation tracking
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_session_id: str = ""
    last_interaction: datetime = field(default_factory=datetime.now)
    message_count: int = 0  # Track messages for analytics

    # App navigation state
    current_task: str = ""  # appointment_booking, medicine_search, etc.
    task_context: Dict[str, Any] = field(default_factory=dict)

    # Progress tracking
    appointment_status: Dict[str, Any] = field(default_factory=dict)  # Track appointment progress
    prescription_summary: Dict[str, Any] = field(default_factory=dict)  # Store prescription summaries
    last_appointment_date: Optional[datetime] = None
    post_appointment_feedback_pending: bool = False
    medicine_reminders: List[Dict[str, Any]] = field(default_factory=list)

    # Interactive UI state
    show_appointment_button: bool = False
    show_medicine_scan_button: bool = False
    show_prescription_button: bool = False
    pending_actions: List[str] = field(default_factory=list)

    # User preferences
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    emergency_contact: Dict[str, str] = field(default_factory=dict)

    # Usage statistics
    total_conversations: int = 0
    total_appointments_booked: int = 0
    total_health_records_accessed: int = 0
    appointments_completed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            'user_id': self.user_id,
            'patient_id': self.patient_id,
            'full_name': self.full_name,
            'preferred_language': self.preferred_language,
            'location': self.location,
            'conversation_history': self.conversation_history[-10:],  # Keep last 10 turns
            'current_session_id': self.current_session_id,
            'last_interaction': self.last_interaction.isoformat(),
            'message_count': self.message_count,
            'current_task': self.current_task,
            'task_context': self.task_context,
            'appointment_status': self.appointment_status,
            'prescription_summary': self.prescription_summary,
            'last_appointment_date': self.last_appointment_date.isoformat() if self.last_appointment_date else None,
            'post_appointment_feedback_pending': self.post_appointment_feedback_pending,
            'medicine_reminders': self.medicine_reminders,
            'show_appointment_button': self.show_appointment_button,
            'show_medicine_scan_button': self.show_medicine_scan_button,
            'show_prescription_button': self.show_prescription_button,
            'pending_actions': self.pending_actions,
            'notification_preferences': self.notification_preferences,
            'emergency_contact': self.emergency_contact,
            'total_conversations': self.total_conversations,
            'total_appointments_booked': self.total_appointments_booked,
            'total_health_records_accessed': self.total_health_records_accessed,
            'appointments_completed': self.appointments_completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create profile from dictionary"""
        profile = cls(
            user_id=data.get('user_id', ''),
            patient_id=data.get('patient_id', ''),
            full_name=data.get('full_name', ''),
            preferred_language=data.get('preferred_language', 'hi'),
            location=data.get('location', ''),
            conversation_history=data.get('conversation_history', []),
            current_session_id=data.get('current_session_id', ''),
            message_count=data.get('message_count', 0),
            current_task=data.get('current_task', ''),
            task_context=data.get('task_context', {}),
            appointment_status=data.get('appointment_status', {}),
            prescription_summary=data.get('prescription_summary', {}),
            medicine_reminders=data.get('medicine_reminders', []),
            show_appointment_button=data.get('show_appointment_button', False),
            show_medicine_scan_button=data.get('show_medicine_scan_button', False),
            show_prescription_button=data.get('show_prescription_button', False),
            pending_actions=data.get('pending_actions', []),
            notification_preferences=data.get('notification_preferences', {}),
            emergency_contact=data.get('emergency_contact', {}),
            total_conversations=data.get('total_conversations', 0),
            total_appointments_booked=data.get('total_appointments_booked', 0),
            total_health_records_accessed=data.get('total_health_records_accessed', 0),
            appointments_completed=data.get('appointments_completed', 0)
        )

        # Parse last_interaction
        last_interaction_str = data.get('last_interaction')
        if last_interaction_str:
            try:
                profile.last_interaction = datetime.fromisoformat(last_interaction_str)
            except:
                profile.last_interaction = datetime.now()

        # Parse last_appointment_date
        last_appointment_str = data.get('last_appointment_date')
        if last_appointment_str:
            try:
                profile.last_appointment_date = datetime.fromisoformat(last_appointment_str)
            except:
                profile.last_appointment_date = None

        # Parse post_appointment_feedback_pending
        profile.post_appointment_feedback_pending = data.get('post_appointment_feedback_pending', False)

        return profile

class ProgressiveConversationMemory:
    """
    Simplified conversation memory system for Sehat Sahara Health Assistant.
    Focuses on task context and user preferences rather than complex mental health tracking.
    """
    
    def __init__(self, max_history_per_user: int = 50):
        self.logger = logging.getLogger(__name__)
        self.max_history_per_user = max_history_per_user
        
        # In-memory storage (in production, this would be backed by database)
        self.user_profiles: Dict[str, UserProfile] = {}
        self.session_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Task state tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Simple analytics
        self.conversation_stats = defaultdict(int)
        
        self.logger.info("✅ Sehat Sahara Conversation Memory initialized")
    
    def create_or_get_user(self, user_id: str, **kwargs) -> UserProfile:
        """Create or retrieve user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                patient_id=kwargs.get('patient_id', ''),
                full_name=kwargs.get('full_name', ''),
                preferred_language=kwargs.get('preferred_language', 'hi'),
                location=kwargs.get('location', '')
            )
            self.logger.info(f"Created new user profile for: {user_id}")
        else:
            # Update existing profile with new information
            profile = self.user_profiles[user_id]
            for key, value in kwargs.items():
                if hasattr(profile, key) and value:
                    setattr(profile, key, value)
        
        return self.user_profiles[user_id]
    
    def add_conversation_turn(self,
                             user_id: str,
                             user_message: str,
                             bot_response: str,
                             nlu_result: Dict[str, Any],
                             action_taken: str = None,
                             session_id: str = None) -> None:
        """Add a conversation turn to user's history"""
        
        profile = self.create_or_get_user(user_id)
        
        # Create conversation turn
        turn = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'intent': nlu_result.get('primary_intent', 'unknown'),
            'language': nlu_result.get('language_detected', 'hi'),
            'urgency_level': nlu_result.get('urgency_level', 'low'),
            'action_taken': action_taken,
            'session_id': session_id or profile.current_session_id
        }
        
        # Add to conversation history
        profile.conversation_history.append(turn)
        
        # Keep only recent history
        if len(profile.conversation_history) > self.max_history_per_user:
            profile.conversation_history = profile.conversation_history[-self.max_history_per_user:]
        
        # Update profile statistics
        profile.total_conversations += 1
        profile.last_interaction = datetime.now()
        profile.message_count += 1

        
        # Update session context
        if session_id:
            profile.current_session_id = session_id
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = {
                    'user_id': user_id,
                    'start_time': datetime.now(),
                    'turns_count': 0,
                    'actions_taken': []
                }
            
            self.session_contexts[session_id]['turns_count'] += 1
            if action_taken:
                self.session_contexts[session_id]['actions_taken'].append(action_taken)
        
        # Update conversation statistics
        intent = nlu_result.get('primary_intent', 'unknown')
        self.conversation_stats[f'intent_{intent}'] += 1
        self.conversation_stats['total_conversations'] += 1
        
        self.logger.debug(f"Added conversation turn for user {user_id}: {intent}")
    
    def get_conversation_context(self, user_id: str, turns: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context for a user"""
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        return profile.conversation_history[-turns:] if profile.conversation_history else []
    
    def set_current_task(self, user_id: str, task: str, context: Dict[str, Any] = None) -> None:
        """Set current task for user (e.g., appointment booking flow)"""
        profile = self.create_or_get_user(user_id)
        profile.current_task = task
        profile.task_context = context or {}
        
        # Track active task
        self.active_tasks[user_id] = {
            'task': task,
            'context': context or {},
            'started_at': datetime.now(),
            'status': 'active'
        }
        
        self.logger.info(f"Set current task for user {user_id}: {task}")
    
    def get_current_task(self, user_id: str) -> Dict[str, Any]:
        """Get current task and context for user"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            return {
                'task': profile.current_task,
                'context': profile.task_context
            }
        return {'task': '', 'context': {}}
    
    def complete_task(self, user_id: str, task_result: Dict[str, Any] = None) -> None:
        """Mark current task as completed"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            completed_task = profile.current_task
            
            # Update statistics based on completed task
            if completed_task == 'appointment_booking':
                profile.total_appointments_booked += 1
            elif completed_task == 'health_record_request':
                profile.total_health_records_accessed += 1
            
            # Clear current task
            profile.current_task = ""
            profile.task_context = {}
            
            # Update active tasks
            if user_id in self.active_tasks:
                self.active_tasks[user_id]['status'] = 'completed'
                self.active_tasks[user_id]['completed_at'] = datetime.now()
                self.active_tasks[user_id]['result'] = task_result or {}
            
            self.logger.info(f"Completed task for user {user_id}: {completed_task}")
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Update user preferences"""
        profile = self.create_or_get_user(user_id)
        
        # Update language preference
        if 'language' in preferences:
            profile.preferred_language = preferences['language']
        
        # Update notification preferences
        if 'notifications' in preferences:
            profile.notification_preferences.update(preferences['notifications'])
        
        # Update emergency contact
        if 'emergency_contact' in preferences:
            profile.emergency_contact.update(preferences['emergency_contact'])
        
        # Update location
        if 'location' in preferences:
            profile.location = preferences['location']
        
        self.logger.info(f"Updated preferences for user {user_id}")
    
    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user summary"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        recent_intents = []
        
        # Get recent intents from conversation history
        for turn in profile.conversation_history[-5:]:
            if turn.get('intent'):
                recent_intents.append(turn['intent'])
        
        return {
            'user_info': {
                'user_id': profile.user_id,
                'patient_id': profile.patient_id,
                'full_name': profile.full_name,
                'preferred_language': profile.preferred_language,
                'location': profile.location
            },
            'current_state': {
                'current_task': profile.current_task,
                'task_context': profile.task_context,
                'last_interaction': profile.last_interaction.isoformat()
            },
            'usage_stats': {
                'total_conversations': profile.total_conversations,
                'total_appointments_booked': profile.total_appointments_booked,
                'total_health_records_accessed': profile.total_health_records_accessed,
                'recent_intents': recent_intents
            },
            'preferences': {
                'language': profile.preferred_language,
                'notifications': profile.notification_preferences,
                'emergency_contact': profile.emergency_contact
            },
            'progress': {
                'appointment_status': profile.appointment_status,
                'prescription_count': len(profile.prescription_summary),
                'last_appointment': profile.last_appointment_date.isoformat() if profile.last_appointment_date else None,
                'appointments_booked': profile.total_appointments_booked,
                'appointments_completed': profile.appointments_completed,
                'feedback_pending': profile.post_appointment_feedback_pending,
                'medicine_reminders_count': len(profile.medicine_reminders)
            },
            'interactive_state': {
                'show_appointment_button': profile.show_appointment_button,
                'show_medicine_scan_button': profile.show_medicine_scan_button,
                'show_prescription_button': profile.show_prescription_button,
                'pending_actions': profile.pending_actions,
                'message_count': profile.message_count
            }
        }
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for a specific session"""
        return self.session_contexts.get(session_id, {})
    
    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """Clean up old session contexts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        old_sessions = []
        
        for session_id, context in self.session_contexts.items():
            if context.get('start_time', datetime.now()) < cutoff_time:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            del self.session_contexts[session_id]
        
        self.logger.info(f"Cleaned up {len(old_sessions)} old sessions")
        return len(old_sessions)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide conversation statistics"""
        active_users = len([p for p in self.user_profiles.values() 
                           if (datetime.now() - p.last_interaction).days < 7])
        
        total_tasks_completed = sum([
            p.total_appointments_booked + p.total_health_records_accessed 
            for p in self.user_profiles.values()
        ])
        
        return {
            'total_users': len(self.user_profiles),
            'active_users_week': active_users,
            'total_conversations': self.conversation_stats.get('total_conversations', 0),
            'total_tasks_completed': total_tasks_completed,
            'active_sessions': len(self.session_contexts),
            'conversation_stats': dict(self.conversation_stats),
            'supported_languages': ['hi', 'pa', 'en']
        }
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data (for privacy compliance)"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        return {
            'profile': profile.to_dict(),
            'active_task': self.active_tasks.get(user_id, {}),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data (for privacy compliance)"""
        try:
            if user_id in self.user_profiles:
                del self.user_profiles[user_id]
            
            if user_id in self.active_tasks:
                del self.active_tasks[user_id]
            
            # Remove from session contexts
            sessions_to_remove = [
                session_id for session_id, context in self.session_contexts.items()
                if context.get('user_id') == user_id
            ]
            
            for session_id in sessions_to_remove:
                del self.session_contexts[session_id]
            
            self.logger.info(f"Deleted all data for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting user data for {user_id}: {e}")
            return False
    
    def save_to_file(self, filepath: str) -> bool:
        """Save conversation memory to file"""
        try:
            data = {
                'user_profiles': {uid: profile.to_dict() for uid, profile in self.user_profiles.items()},
                'session_contexts': self.session_contexts,
                'active_tasks': self.active_tasks,
                'conversation_stats': dict(self.conversation_stats),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Conversation memory saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving conversation memory: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load conversation memory from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load user profiles
            self.user_profiles = {}
            for uid, profile_data in data.get('user_profiles', {}).items():
                self.user_profiles[uid] = UserProfile.from_dict(profile_data)

            # Load other data
            self.session_contexts = data.get('session_contexts', {})
            self.active_tasks = data.get('active_tasks', {})
            self.conversation_stats = defaultdict(int, data.get('conversation_stats', {}))

            self.logger.info(f"Conversation memory loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading conversation memory: {e}")
            return False

    def update_appointment_status(self, user_id: str, appointment_id: str, status: str, appointment_date: datetime = None) -> None:
        """Update appointment status and track progress"""
        profile = self.create_or_get_user(user_id)

        profile.appointment_status[appointment_id] = {
            'status': status,
            'appointment_date': appointment_date.isoformat() if appointment_date else None,
            'updated_at': datetime.now().isoformat()
        }

        if status == 'completed' and appointment_date:
            profile.last_appointment_date = appointment_date
            profile.appointments_completed += 1
            profile.post_appointment_feedback_pending = True

        self.logger.info(f"Updated appointment {appointment_id} status for user {user_id}: {status}")

    def check_post_appointment_followup(self, user_id: str) -> Dict[str, Any]:
        """Check if post-appointment follow-up is needed"""
        if user_id not in self.user_profiles:
            return {'needed': False}

        profile = self.user_profiles[user_id]

        if not profile.post_appointment_feedback_pending:
            return {'needed': False}

        # Check if it's been at least 1 day since appointment
        if profile.last_appointment_date:
            days_since_appointment = (datetime.now() - profile.last_appointment_date).days
            if days_since_appointment >= 1:
                return {
                    'needed': True,
                    'appointment_date': profile.last_appointment_date.isoformat(),
                    'days_since': days_since_appointment
                }

        return {'needed': False}

    def complete_post_appointment_feedback(self, user_id: str) -> None:
        """Mark post-appointment feedback as completed"""
        if user_id in self.user_profiles:
            self.user_profiles[user_id].post_appointment_feedback_pending = False
            self.logger.info(f"Completed post-appointment feedback for user {user_id}")

    def get_user_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive progress summary for user"""
        if user_id not in self.user_profiles:
            return {}

        profile = self.user_profiles[user_id]

        return {
            'appointment_history': profile.appointment_status,
            'prescription_count': len(profile.prescription_summary),
            'last_appointment': profile.last_appointment_date.isoformat() if profile.last_appointment_date else None,
            'appointments_booked': profile.total_appointments_booked,
            'appointments_completed': profile.appointments_completed,
            'feedback_pending': profile.post_appointment_feedback_pending,
            'interactive_buttons': {
                'appointment': profile.show_appointment_button,
                'medicine_scan': profile.show_medicine_scan_button,
                'prescription': profile.show_prescription_button
            },
            'pending_actions': profile.pending_actions,
            'medicine_reminders_count': len(profile.medicine_reminders),
            'prescription_summary_available': len(profile.prescription_summary) > 0
        }

    def schedule_medicine_reminder(self, user_id: str, medicine_data: Dict[str, Any]) -> None:
        """Schedule medicine reminders for user"""
        profile = self.create_or_get_user(user_id)

        reminder = {
            'medicine_name': medicine_data['name'],
            'dosage': medicine_data['dosage'],
            'frequency': medicine_data['frequency'],
            'times': medicine_data['times'],
            'duration_days': medicine_data['duration_days'],
            'start_date': medicine_data['start_date'],
            'instructions': medicine_data.get('instructions', ''),
            'reminder_enabled': True,
            'created_at': datetime.now().isoformat()
        }

        profile.medicine_reminders.append(reminder)
        self.logger.info(f"Added medicine reminder for user {user_id}: {medicine_data['name']}")

    def get_medicine_reminders(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active medicine reminders for user"""
        if user_id not in self.user_profiles:
            return []

        profile = self.user_profiles[user_id]
        return [r for r in profile.medicine_reminders if r.get('reminder_enabled', True)]

    def update_reminder_adherence(self, user_id: str, medicine_name: str, taken_time: str) -> None:
        """Update medicine adherence tracking"""
        if user_id not in self.user_profiles:
            return

        profile = self.user_profiles[user_id]
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Track adherence (simplified version)
        self.logger.info(f"Medicine taken: {medicine_name} at {taken_time} for user {user_id}")

    def get_reminder_alerts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending medicine reminders for today"""
        reminders = self.get_medicine_reminders(user_id)
        alerts = []

        current_time = datetime.now().strftime('%H:%M')

        for reminder in reminders:
            for time_slot in reminder.get('times', []):
                if time_slot <= current_time:
                    alerts.append({
                        'medicine_name': reminder['medicine_name'],
                        'dosage': reminder['dosage'],
                        'time': time_slot,
                        'instructions': reminder.get('instructions', '')
                    })

        return alerts
    # Add these two new functions inside the ProgressiveConversationMemory class

    def add_prescription_summary(self, user_id: str, prescription_data: Dict[str, Any]) -> None:
        """
        Adds a new prescription summary to the user's profile.
        This is called after a prescription is uploaded and analyzed.
        """
        profile = self.create_or_get_user(user_id)
        
        # Use the prescription_id from the data as the key
        prescription_id = prescription_data.get('prescription_id', f'rx_{datetime.now().isoformat()}')
        
        # Add a timestamp to the data for sorting later
        prescription_data['saved_at'] = datetime.now().isoformat()
        
        profile.prescription_summary[prescription_id] = prescription_data
        
        self.logger.info(f"Added prescription summary {prescription_id} for user {user_id}")

        # --- Add this line to trigger the automatic reminder creation ---
        self._auto_generate_reminders_from_prescription(profile, prescription_data)
        
        # You can add logic here for auto-generating reminders from this data
        # For now, simply saving it is the critical step.

    def get_prescription_summary(self, user_id: str, prescription_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves a prescription summary for a user.
        If no ID is given, it returns the most recent one.
        """
        profile = self.create_or_get_user(user_id)
        
        if not profile.prescription_summary:
            self.logger.warning(f"No prescription summaries found for user {user_id}")
            return None
        
        if prescription_id and prescription_id in profile.prescription_summary:
            return profile.prescription_summary[prescription_id]
        
        # If no ID is given, find the most recently saved summary
        if not prescription_id:
            try:
                # Sort summaries by the 'saved_at' timestamp in descending order and get the first one
                latest_summary = sorted(
                    profile.prescription_summary.values(), 
                    key=lambda x: x.get('saved_at', '1970-01-01'), 
                    reverse=True
                )[0]
                return latest_summary
            except (IndexError, KeyError):
                # Fallback in case sorting fails or 'saved_at' is missing
                self.logger.warning(f"Could not sort prescriptions for user {user_id}, returning first available.")
                return next(iter(profile.prescription_summary.values()))

        self.logger.warning(f"Prescription ID {prescription_id} not found for user {user_id}")
        return None
    
    def _auto_generate_reminders_from_prescription(self, profile: UserProfile, prescription_data: Dict[str, Any]):
        """
        A helper method to intelligently parse prescription data and create
        pre-filled, disabled reminders for the user to activate.
        """
        medications = prescription_data.get('medications', [])
        if not medications:
            return

        self.logger.info(f"Attempting to auto-generate reminders from {len(medications)} medications for user {profile.user_id}.")
        
        new_reminders_created = 0

        for med in medications:
            med_name = med.get('name')
            if not med_name:
                continue

            # --- Check to prevent adding duplicate reminders for the same medicine ---
            existing_reminder_names = {r.get('medicine_name', '').lower() for r in profile.medicine_reminders}
            if med_name.lower() in existing_reminder_names:
                self.logger.info(f"Reminder for '{med_name}' already exists. Skipping.")
                continue

            # --- Intelligent Time Parsing Logic ---
            times = []
            time_instruction = med.get('time', '').lower()
            
            # Map common phrases to specific times
            if 'once a day' in time_instruction or 'daily' in time_instruction or 'morning' in time_instruction:
                times.append("08:00")
            if 'twice a day' in time_instruction or 'afternoon' in time_instruction:
                times.append("14:00")
            if 'thrice a day' in time_instruction or 'evening' in time_instruction:
                times.append("20:00")
            if 'night' in time_instruction or 'before bed' in time_instruction:
                times.append("22:00")
            
            # If no phrases match, but there are multiple times, try to extract them
            if not times and ',' in time_instruction:
                potential_times = [t.strip() for t in time_instruction.split(',')]
                times.extend(potential_times)
            
            # Default if no time is found
            if not times:
                times.append("09:00") # Default to a morning reminder

            reminder = {
                'medicine_name': med_name,
                'dosage': med.get('dosage', 'As prescribed'),
                'times': times,
                'duration_days': 30,  # Default duration
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'instructions': med.get('time', 'As directed by your doctor.'),
                'source': 'prescription_upload', # To identify auto-generated reminders
                'reminder_enabled': False, # User must manually enable it
                'created_at': datetime.now().isoformat()
            }

            profile.medicine_reminders.append(reminder)
            new_reminders_created += 1
            self.logger.info(f"Successfully created a pre-filled reminder for '{med_name}'.")

        if new_reminders_created > 0:
            self.logger.info(f"Auto-generated {new_reminders_created} new medicine reminders for user {profile.user_id}.")

# Global instance for easy import
conversation_memory = ProgressiveConversationMemory()
