"""
Sehat Sahara Health Assistant NLU Processor
Natural Language Understanding for Health App Navigation and Task-Oriented Commands
Supports Punjabi, Hindi, and English for rural patients
"""

import os
import pickle
import logging
import re
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import threading
import requests
import time


class ProgressiveNLUProcessor:
    """
    NLU processor for Sehat Sahara Health Assistant with multilingual support.
    Processes user commands for health app navigation and task completion.
    """

    def __init__(self, model_path: str = None, api_keys: str = None):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        # --- UPDATE THIS SECTION ---
        # Use a different environment variable for clarity, e.g., NLU_API_KEYS
        # It reads multiple keys separated by commas
        self.api_keys = [key.strip() for key in os.getenv('NLU_API_KEYS', '').split(',') if key.strip()]
        self.openrouter_base_url = "https://api.groq.com/openai/v1"
        self.openrouter_model = "llama-3.1-8b-instant"

        # Key rotation state
        self.current_key_index = 0
        self.last_switch_time = 0
        # --- END OF UPDATE ---

        # Initialize OpenRouter for enhanced understanding
        self.openrouter_available = False
        self.use_openrouter = False

        if self.api_keys:
            try:
                # Test API key validity with improved error handling
                connection_result = self._test_openrouter_connection()
                if connection_result:
                    self.use_openrouter = True
                    self.openrouter_available = True
                    self.logger.info("✅ NLU API connected for enhanced multilingual NLU")
                else:
                    self.logger.warning("❌ NLU API key validation failed - check your API key")
                    self.logger.info("💡 To get a valid API key, visit: https://openrouter.ai/keys")
            except Exception as e:
                self.logger.warning(f"Could not connect to Groq API: {e}")
                self.logger.info("🔧 The system will work with enhanced keyword-based classification")
        else:
            self.logger.warning("⚠️ No NLU API key provided. Enhanced NLU features will be limited.")
            self.logger.info("💡 Set NLU_API_KEY environment variable to enable AI-powered features")

        # Legacy semantic model fallback (optional)
        self.sentence_model = None
        self.use_semantic = False

        # Function/button display configuration
        self.function_display_threshold = 0.7  # Minimum confidence to show function buttons
        self.button_display_rules = {
            'appointment_booking': ['book_appointment', 'call_doctor', 'emergency_contact'],
            'medicine_scan': ['scan_medicine', 'search_pharmacy', 'get_directions'],
            'prescription_inquiry': ['view_prescription', 'set_reminder', 'refill_request'],
            'emergency_assistance': ['call_ambulance', 'call_emergency_contact', 'show_nearby_hospitals']
        }

        # Health app intent categories with multilingual keywords
        self.intent_categories = {
            'appointment_booking': {
                'keywords': [
                    # English - expanded
                    'book appointment', 'need to see doctor', 'doctor appointment', 'schedule appointment',
                    'meet doctor', 'consultation', 'book doctor', 'see doctor', 'doctor visit',
                    'make appointment', 'fix appointment', 'get appointment', 'appointment booking',
                    'want to meet doctor', 'need doctor consultation', 'doctor consultation',
                    # Hindi (Latin script) - greatly expanded
                    'doctor se milna hai', 'appointment book karni hai', 'doctor ko dikhana hai',
                    'doctor ke paas jana hai', 'appointment chahiye', 'doctor se baat karni hai',
                    'appointment banao', 'appointment fix karo', 'doctor se milo', 'doctor ko dikhao',
                    'appointment karvao', 'doctor appointment chahiye', 'doctor ke paas jao',
                    'appointment book karo', 'doctor mila do', 'appointment de do',
                    # Hindi variations with English combinations
                    'doctor appointment book', 'appointment book hindi', 'doctor se milna book',
                    'appointment chahiye doctor', 'doctor ko dikhana appointment',
                    # Punjabi (Latin script) - greatly expanded
                    'doctor nu milna hai', 'appointment book karni hai', 'doctor kol jana hai',
                    'doctor nu dikhana hai', 'doctor de kol appointment', 'vaid nu milna hai',
                    'appointment banao', 'appointment fix karo', 'doctor nu milo', 'doctor nu dikhao',
                    'appointment karvao', 'doctor appointment chahida', 'doctor kol jao',
                    'appointment book karo', 'doctor mila deo', 'appointment de deo',
                    # Punjabi variations with English combinations
                    'doctor appointment book', 'appointment book punjabi', 'doctor nu milna book',
                    'appointment chahida doctor', 'doctor nu dikhana appointment'
                ],
                'urgency_indicators': ['urgent', 'emergency', 'turant', 'jaldi', 'emergency hai', 'urgent hai']
            },
            'appointment_view': {
                'keywords': [
                    # English
                    'my appointments', 'when is my appointment', 'next appointment', 'appointment time',
                    'show appointments', 'check appointment', 'appointment details',
                    # Hindi (Latin script)
                    'meri appointment kab hai', 'appointment ka time', 'appointment dekhni hai',
                    'kab hai appointment', 'appointment ki jankari',
                    # Punjabi (Latin script)
                    'meri appointment kado hai', 'appointment kado hai', 'appointment dekhan hai',
                    'appointment da time', 'appointment di jankari'
                ]
            },
            'appointment_cancel': {
                'keywords': [
                    # English
                    'cancel appointment', 'cancel my appointment', 'dont want appointment',
                    'remove appointment', 'delete appointment',
                    # Hindi (Latin script)
                    'appointment cancel karni hai', 'appointment nahi chahiye', 'appointment cancel karo',
                    'appointment hatana hai',
                    # Punjabi (Latin script)
                    'appointment cancel karni hai', 'appointment nahi chahidi', 'appointment cancel karo',
                    'appointment hatana hai'
                ]
            },
            'health_record_request': {
                'keywords': [
                    # English
                    'my reports', 'blood report', 'test results', 'medical records', 'health records',
                    'last report', 'show my reports', 'medical history', 'prescription history',
                    # Hindi (Latin script)
                    'meri report', 'blood report', 'test ka result', 'medical record',
                    'pichli report', 'dawai ki history', 'report dikhao',
                    # Punjabi (Latin script)
                    'meri report', 'blood report', 'test da result', 'medical record',
                    'pichli report', 'dawai di history', 'report dikhao'
                ]
            },
            'symptom_triage': {
                'keywords': [
                    # English
                    'fever', 'headache', 'pain', 'cough', 'cold', 'stomach pain', 'chest pain',
                    'feeling sick', 'not feeling well', 'symptoms', 'body ache', 'chills',
                    'vomiting', 'diarrhea', 'jaundice', 'yellow skin', 'persistent cough',
                    'weight loss', 'fatigue', 'weakness', 'high fever', 'severe headache',
                    'joint pain', 'skin rash', 'abdominal pain', 'nausea', 'dehydration',
                    # Hindi (Latin script)
                    'bukhar hai', 'sir dard hai', 'dard hai', 'khansi hai', 'pet dard hai',
                    'tabiyat kharab hai', 'bimari hai', 'body pain hai', 'chill lag rahi hai',
                    'ulti ho rahi hai', 'dast aa rahe hain', 'piliya hai', 'peeli chamdi',
                    'khansi nahi rukti', 'vajan kam ho raha hai', 'thakan mahsus ho rahi hai',
                    'kamzori hai', 'tez bukhar', 'tez sir dard', 'joint pain', 'chamdi par rash',
                    'pet mein dard', 'ghabrahat', 'dehydration',
                    # Punjabi (Latin script)
                    'bukhar hai', 'sir dukh raha hai', 'dard hai', 'khansi hai', 'pet dukh raha hai',
                    'tabiyat kharab hai', 'bimari hai', 'body pain hai', 'chill lag rahi hai',
                    'ulti ho rahi hai', 'dast aa rahe hain', 'piliya hai', 'peeli chamdi',
                    'khansi nahi rukdi', 'vajan kam ho raha hai', 'thakan mahsus ho rahi hai',
                    'kamzori hai', 'tez bukhar', 'tez sir dard', 'joint pain', 'chamdi te rash',
                    'pet vich dard', 'ghabrahat', 'dehydration'
                ],
                'urgency_indicators': ['severe pain', 'chest pain', 'breathing problem', 'emergency', 'accident', 'high fever', 'unconscious', 'severe vomiting', 'blood in stool']
            },
            'find_medicine': {
                'keywords': [
                    # English
                    'find medicine', 'where to buy medicine', 'pharmacy near me', 'medicine shop',
                    'buy medicine', 'medicine available', 'find pharmacy',
                    # Hindi (Latin script)
                    'dawai kahan milegi', 'medicine shop', 'pharmacy', 'dawai leni hai',
                    'medicine kahan hai', 'dawai ki dukan',
                    # Punjabi (Latin script)
                    'dawai kithe milegi', 'medicine shop', 'pharmacy', 'dawai leni hai',
                    'medicine kithe hai', 'dawai di dukan'
                ]
            },
            'prescription_inquiry': {
                'keywords': [
                    # English
                    'how to take medicine', 'medicine dosage', 'when to take', 'medicine instructions',
                    'tablet kitni', 'medicine timing', 'prescription details', 'upload prescription',
                    'view prescription', 'prescription upload', 'show prescription', 'add prescription',
                    'prescription image', 'doctor prescription', 'medical prescription',
                    # Hindi (Latin script) - expanded
                    'dawai kaise leni hai', 'kitni tablet leni hai', 'dawai ka time',
                    'medicine kab leni hai', 'dawai ki jankari', 'dawai upload karo',
                    'dawai ki parchi upload', 'dawai ki parchi dekho', 'dawai ki parchi',
                    'doctor ki parchi', 'prescription upload', 'prescription dikhao',
                    'dawai ki photo', 'dawai ka paper', 'dawai ki slip',
                    # Punjabi (Latin script) - expanded
                    'dawai kive leni hai', 'kinni tablet leni hai', 'dawai da time',
                    'medicine kado leni hai', 'dawai di jankari', 'dawai upload karo',
                    'dawai di parchi upload', 'dawai di parchi dekho', 'dawai di parchi',
                    'doctor di parchi', 'prescription upload', 'prescription dikhao',
                    'dawai di photo', 'dawai da paper', 'dawai di slip'
                ]
            },
            'medicine_scan': {
                'keywords': [
                    # English - expanded
                    'scan medicine', 'check medicine', 'medicine scanner', 'identify medicine',
                    'what is this medicine', 'medicine name', 'scan medicine packaging',
                    'medicine identification', 'scan tablet', 'scan pill', 'medicine scan',
                    'scan medicine box', 'medicine label scan', 'scan medicine strip',
                    # Hindi (Latin script) - greatly expanded
                    'medicine scan karo', 'ye kya dawai hai', 'medicine check karo',
                    'dawai ka naam', 'medicine identify karo', 'dawai scan karo',
                    'dawai check karo', 'ye kya medicine hai', 'medicine scan karni hai',
                    'dawai ki scanning', 'medicine ki pehchan', 'dawai identify karo',
                    'dawai ka naam batao', 'medicine scan karna hai', 'dawai dekhna hai',
                    # Hindi variations with English combinations
                    'medicine scan hindi', 'dawai scan english', 'medicine check karna',
                    'dawai scan karo na', 'medicine scan chahiye', 'dawai scan de do',
                    # Punjabi (Latin script) - greatly expanded
                    'medicine scan karo', 'eh ki dawai hai', 'medicine check karo',
                    'dawai da naam', 'medicine identify karo', 'dawai scan karo',
                    'dawai check karo', 'eh ki medicine hai', 'medicine scan karni hai',
                    'dawai di scanning', 'medicine di pehchan', 'dawai identify karo',
                    'dawai da naam dasso', 'medicine scan karna hai', 'dawai vekhan hai',
                    # Punjabi variations with English combinations
                    'medicine scan punjabi', 'dawai scan english', 'medicine check karna',
                    'dawai scan karo na', 'medicine scan chahida', 'dawai scan de deo'
                ]
            },
            'emergency_assistance': {
                'keywords': [
                    # English
                    'emergency', 'help me', 'accident', 'urgent help', 'ambulance',
                    'emergency call', 'immediate help', 'crisis',
                    # Hindi (Latin script)
                    'emergency hai', 'help karo', 'accident hua hai', 'ambulance chahiye',
                    'turant help chahiye', 'emergency call',
                    # Punjabi (Latin script)
                    'emergency hai', 'help karo', 'accident ho gaya hai', 'ambulance chahida',
                    'turant help chahidi', 'emergency call'
                ],
                'urgency_indicators': ['emergency', 'accident', 'ambulance', 'urgent', 'help']
            },
            'report_issue': {
                'keywords': [
                    # English
                    'complaint', 'doctor was rude', 'overcharged', 'bad service', 'report problem',
                    'feedback', 'issue with', 'problem with',
                    # Hindi (Latin script)
                    'complaint hai', 'doctor rude tha', 'zyada paisa liya', 'service kharab thi',
                    'problem hai', 'shikayat hai',
                    # Punjabi (Latin script)
                    'complaint hai', 'doctor rude si', 'zyada paisa liya', 'service kharab si',
                    'problem hai', 'shikayat hai'
                ]
            },
            'general_inquiry': {
                'keywords': [
                    # English
                    'how to use app', 'help', 'what can you do', 'app features',
                    'how does this work', 'guide me', 'tutorial',
                    # Hindi (Latin script)
                    'app kaise use kare', 'help chahiye', 'app ki features',
                    'kaise kaam karta hai', 'guide karo',
                    # Punjabi (Latin script)
                    'app kive use karna hai', 'help chahidi', 'app dian features',
                    'kive kaam karda hai', 'guide karo'
                ]
            },
            'how_to_appointment_booking': {
                'keywords': [
                    # English
                    'how to book appointment', 'how do i book appointment', 'appointment kaise book karu',
                    'how to schedule appointment', 'appointment kaise karu', 'how to make appointment',
                    # Hindi (Latin script)
                    'appointment kaise book karu', 'appointment kaise karu', 'appointment kaise banaun',
                    'appointment kaise fix karu', 'appointment kaise schedule karu', 'doctor kaise book karu',
                    'doctor se kaise milu', 'appointment kaise chahiye',
                    # Punjabi (Latin script)
                    'appointment kaise book karu', 'appointment kaise karu', 'appointment kaise banaun',
                    'appointment kaise fix karu', 'appointment kaise schedule karu', 'doctor kaise book karu',
                    'doctor naal kaise milu', 'appointment kaise chahida'
                ]
            },
            'how_to_medicine_scan': {
                'keywords': [
                    # English
                    'how to scan medicine', 'how do i scan medicine', 'medicine kaise scan karu',
                    'how to identify medicine', 'medicine kaise identify karu', 'how to check medicine',
                    # Hindi (Latin script)
                    'medicine kaise scan karu', 'dawai kaise scan karu', 'medicine kaise check karu',
                    'dawai kaise identify karu', 'medicine kaise pehchanu', 'dawai kaise pehchanu',
                    'medicine scanner kaise use karu', 'dawai scanner kaise use karu',
                    # Punjabi (Latin script)
                    'medicine kaise scan karu', 'dawai kaise scan karu', 'medicine kaise check karu',
                    'dawai kaise identify karu', 'medicine kaise pehchanu', 'dawai kaise pehchanu',
                    'medicine scanner kaise use karu', 'dawai scanner kaise use karu'
                ]
            },
            'how_to_prescription_upload': {
                'keywords': [
                    # English
                    'how to upload prescription', 'how do i upload prescription', 'prescription kaise upload karu',
                    'how to add prescription', 'prescription kaise add karu', 'how to scan prescription',
                    # Hindi (Latin script)
                    'prescription kaise upload karu', 'parchi kaise upload karu', 'prescription kaise add karu',
                    'parchi kaise add karu', 'prescription kaise scan karu', 'parchi kaise scan karu',
                    'dawai ki parchi kaise upload karu', 'doctor ki parchi kaise upload karu',
                    # Punjabi (Latin script)
                    'prescription kaise upload karu', 'parchi kaise upload karu', 'prescription kaise add karu',
                    'parchi kaise add karu', 'prescription kaise scan karu', 'parchi kaise scan karu',
                    'dawai di parchi kaise upload karu', 'doctor di parchi kaise upload karu'
                ]
            },
            'post_appointment_followup': {
                'keywords': [
                    # English
                    'feeling better', 'appointment went well', 'doctor visit good', 'feeling worse',
                    'appointment was helpful', 'doctor helped', 'not feeling better', 'need to see doctor again',
                    'appointment feedback', 'how was appointment', 'doctor consultation',
                    # Hindi (Latin script)
                    'ab accha lag raha hai', 'appointment accha tha', 'doctor ne madad ki', 'ab bhi bura lag raha hai',
                    'appointment se fayda hua', 'doctor ne sahi batai', 'accha nahi lag raha', 'phir doctor ke paas jana hai',
                    'appointment feedback', 'appointment kaisa tha', 'doctor se consultation',
                    # Punjabi (Latin script)
                    'hunn changa lag raha hai', 'appointment changa si', 'doctor ne madad kiti', 'hunn vi bura lag raha hai',
                    'appointment ton fayda hoya', 'doctor ne sahi dassi', 'changa nahi lag raha', 'phir doctor kol jana hai',
                    'appointment feedback', 'appointment kaisa si', 'doctor naal consultation'
                ]
            },
            'prescription_summary_request': {
                'keywords': [
                    # English
                    'what did doctor prescribe', 'show prescription summary', 'medicine list', 'doctor prescription',
                    'what medicines', 'prescription details', 'doctor advice', 'medicine summary',
                    # Hindi (Latin script)
                    'doctor ne kya likha', 'prescription summary dikhao', 'dawai ki list', 'doctor ki prescription',
                    'kya dawaiyan', 'prescription details', 'doctor ki salah', 'dawai summary',
                    # Punjabi (Latin script)
                    'doctor ne ki likhya', 'prescription summary dikhao', 'dawai di list', 'doctor di prescription',
                    'ki dawaiyan', 'prescription details', 'doctor di salah', 'dawai summary'
                ]
            },
            'set_medicine_reminder': {
                'keywords': [
                    # English
                    'set reminder', 'medicine reminder', 'remind me to take medicine', 'schedule medicine',
                    'add medicine reminder', 'medicine alarm', 'pill reminder',
                    # Hindi (Latin script)
                    'reminder set karo', 'dawai ka reminder', 'dawai lene ka reminder', 'medicine reminder',
                    'dawai ka alarm', 'tablet reminder',
                    # Punjabi (Latin script)
                    'reminder set karo', 'dawai da reminder', 'dawai lain da reminder', 'medicine reminder',
                    'dawai da alarm', 'tablet reminder'
                ]
            },
            'prescription_upload': {
                'keywords': [
                    # English
                    'upload prescription', 'add prescription', 'prescription upload', 'scan prescription',
                    'prescription image', 'doctor prescription', 'medical prescription', 'prescription photo',
                    'upload doctor prescription', 'add medical prescription', 'prescription scan',
                    # Hindi (Latin script)
                    'dawai ki parchi upload', 'parchi upload karo', 'dawai ki parchi', 'doctor ki parchi',
                    'parchi add karo', 'parchi scan karo', 'dawai ka paper upload', 'dawai ki slip',
                    'parchi ki photo', 'dawai ki parchi dikhao', 'parchi upload',
                    # Punjabi (Latin script)
                    'dawai di parchi upload', 'parchi upload karo', 'dawai di parchi', 'doctor di parchi',
                    'parchi add karo', 'parchi scan karo', 'dawai da paper upload', 'dawai di slip',
                    'parchi di photo', 'dawai di parchi dikhao', 'parchi upload'
                ]
            },
            'out_of_scope': {
                'keywords': [
                    # English
                    'weather', 'news', 'sports', 'movies', 'music', 'jokes', 'games',
                    'what is 10+10', 'tell me a story', 'sing a song',
                    # Hindi (Latin script)
                    'mausam kaisa hai', 'news kya hai', 'joke sunao', 'gaana gao',
                    'kahani sunao', 'khel',
                    # Punjabi (Latin script)
                    'mausam kaida hai', 'news ki hai', 'joke sunao', 'gana gao',
                    'kahani sunao', 'khel'
                ]
            }
        }

        self.conversation_stages = [
            'initial_contact', 'understanding', 'task_execution', 'confirmation',
            'completion', 'emergency_handling'
        ]

        # Build intent classification data for OpenRouter
        if self.use_openrouter:
            self._prepare_openrouter_intent_data()

        # Load saved model if available
        if model_path and os.path.exists(model_path):
            self.load_nlu_model(model_path)

    # In nlu_processor(2).py

    def _test_openrouter_connection(self) -> bool:
        """Test if the currently active API key is valid and the connection works."""
        try:
            # --- THIS IS THE FIX ---
            # Use self.api_key (singular) to get the single, active API key as a string
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            # --- END OF FIX ---

            # Test with a minimal chat completion request
            payload = {
                "model": self.openrouter_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0.1
            }

            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                self.logger.info("✅ Groq NLU API key validated successfully.")
                return True
            elif response.status_code == 401:
                self.logger.error("❌ Invalid Groq NLU API key - Authentication failed")
                self.logger.info(f"💡 Key ending in ...{self.api_key[-4:]} might be incorrect or lack credits.")
                return False
            else:
                self.logger.warning(f"⚠️ Groq NLU API test returned status {response.status_code}. Key appears valid but service may have issues.")
                return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Groq NLU connection test failed: {e}")
            return False

    def _prepare_openrouter_intent_data(self):
        """Prepare intent classification data for OpenRouter API."""
        try:
            # Create comprehensive intent descriptions for better classification
            self.intent_descriptions = {}

            for intent, data in self.intent_categories.items():
                # Combine keywords into a natural description
                keywords_text = ", ".join(data['keywords'][:10])  # Use top 10 keywords

                # Create multilingual description
                description = f"""
Intent: {intent}
Description: This intent handles user requests related to: {keywords_text}
Common phrases include: {keywords_text}
Urgency indicators: {', '.join(data.get('urgency_indicators', []))}
This is a {'high priority' if 'urgency_indicators' in data else 'standard'} intent for health assistance.
"""
                self.intent_descriptions[intent] = description.strip()

            self.logger.info("✅ OpenRouter intent data prepared for multilingual classification")
        except Exception as e:
            self.logger.error(f"Failed to prepare OpenRouter intent data: {e}")

    # In nlu_processor.py, replace the existing function

    def _call_openrouter_api(self, prompt: str, max_tokens: int = 150, retries: int = None) -> Optional[str]:
        """Make a call to the NLU API with key rotation and retries."""
        if retries is None:
            retries = len(self.api_keys) # Set initial retries to the number of keys

        if not self.use_openrouter or not self.api_key or retries <= 0:
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.openrouter_model,
                "messages": [
                    {"role": "system", "content": "You are an expert NLU system. Respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }

            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 429:
                self.logger.warning(f"NLU API rate limit exceeded for key ...{self.api_key[-4:]}.")
                if self.switch_key():
                    self.logger.info("Retrying NLU request with new key...")
                    return self._call_openrouter_api(prompt, max_tokens, retries - 1)
                else:
                    self.logger.error("NLU: All API keys are rate-limited or unavailable.")
                    return None

            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content")
            else:
                self.logger.error(f"NLU API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"NLU API call failed: {e}")
            return None
    
    # In nlu_processor.py, inside the ProgressiveNLUProcessor class

    @property
    def api_key(self):
        """Returns the currently active API key."""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def switch_key(self):
        """Rotates to the next API key in the list."""
        if time.time() - self.last_switch_time < 10: # 10-second cooldown
            return False

        if len(self.api_keys) > 1:
            old_key_preview = f"...{self.api_key[-4:]}" if self.api_key else "None"
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            new_key_preview = f"...{self.api_key[-4:]}" if self.api_key else "None"
            self.logger.warning(f"NLU: Switching API key from {old_key_preview} to {new_key_preview}")
            self.last_switch_time = time.time()
            return True
        self.logger.warning("NLU: No alternative API keys available to switch to.")
        return False

    def _build_semantic_embeddings(self):
        """Build enhanced semantic embeddings for each intent category using diverse examples"""
        try:
            self.category_embeddings = {}
            self.category_examples = {}  # Store example phrases for each category

            for category, data in self.intent_categories.items():
                # Create diverse example phrases for each category
                examples = []

                # Use keywords to create natural sentences
                keywords = data['keywords'][:8]  # Use more keywords for better representation

                # Create varied sentence patterns
                sentence_patterns = [
                    "I want to {}",
                    "I need to {}",
                    "Can you help me {}",
                    "How do I {}",
                    "I would like to {}",
                    "Please {}",
                    "I need assistance with {}",
                    "Could you {}"
                ]

                for keyword in keywords:
                    # Select random pattern for variety
                    pattern = np.random.choice(sentence_patterns)
                    examples.append(pattern.format(keyword))

                # Add category-specific examples for better representation
                category_specific_examples = self._get_category_specific_examples(category)
                examples.extend(category_specific_examples)

                # Ensure we have enough examples (minimum 5)
                while len(examples) < 5:
                    examples.append(f"Please help with {category.replace('_', ' ')}")

                # Store examples for reference
                self.category_examples[category] = examples[:10]  # Keep top 10

                # Create embeddings for all examples
                embeddings = self.sentence_model.encode(examples)

                # Use mean embedding as category representation
                self.category_embeddings[category] = np.mean(embeddings, axis=0)

            self.logger.info("✅ Enhanced semantic embeddings built for all intent categories")
        except Exception as e:
            self.logger.error(f"Failed to build semantic embeddings: {e}")
            self.use_semantic = False

    def _get_category_specific_examples(self, category: str) -> List[str]:
        """Get category-specific example phrases for better embedding representation"""
        category_examples = {
            'appointment_booking': [
                "I need to schedule a doctor's appointment",
                "Can I book a consultation with a doctor?",
                "I want to make an appointment for tomorrow",
                "Please help me schedule a medical appointment",
                "I need to see a doctor this week"
            ],
            'symptom_triage': [
                "I'm experiencing severe symptoms",
                "I don't feel well and need medical advice",
                "I'm having health issues and need guidance",
                "Can you help assess my symptoms?",
                "I need to understand what these symptoms mean"
            ],
            'medicine_scan': [
                "Can you identify this medicine from the image?",
                "What is the name of this tablet?",
                "Please help me recognize this medication",
                "I need to know what medicine this is",
                "Can you scan and identify this drug?"
            ],
            'emergency_assistance': [
                "This is a medical emergency",
                "I need urgent medical help immediately",
                "Please call emergency services",
                "I'm in a critical situation and need help",
                "Medical emergency - please respond quickly"
            ],
            'prescription_inquiry': [
                "How should I take this medicine?",
                "What are the instructions for this prescription?",
                "When should I take these tablets?",
                "Please explain my medication dosage",
                "I need help understanding my prescription"
            ]
        }
        return category_examples.get(category, [])

    def _ai_powered_classification(self, message: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        AI-powered intent classification using OpenRouter API for multilingual understanding.
        This is the core of the new OpenRouter-based NLU system.
        """
        try:
            if not self.use_openrouter or not hasattr(self, 'intent_descriptions'):
                return None

            # Build context from conversation history
            context_prompt = self._build_conversation_context(conversation_history)

            # Create multilingual classification prompt
            intent_list = "\n".join([
                f"- {intent}: {desc[:100]}..."
                for intent, desc in self.intent_descriptions.items()
            ])

            # In nlu_processor.py, inside the _ai_powered_classification function

            prompt = f"""
Analyze this user message in a health assistance app and classify the intent.

USER MESSAGE: "{message}"
CONVERSATION CONTEXT: {context_prompt}

Available intents:
{intent_list}

Respond with JSON format:
{{
    "primary_intent": "intent_name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "language_detected": "en/hi/pa",
    "urgency_level": "low/medium/high/emergency",
    "multi_intent": ["intent1", "intent2"]
}}

Consider:
- Multilingual support (English, Hindi, Punjabi)
- Health app context
- Urgency indicators
- Conversation flow
- Only return valid JSON

# --- ADD THIS NEW RULE ---
- **CRITICAL RULE:** If the CONVERSATION CONTEXT indicates a 'symptom_triage' task is in progress, and the USER MESSAGE is a short follow-up (e.g., 'since yesterday', 'yes', 'and?', 'constant'), you MUST classify the intent as 'symptom_triage'. Do NOT classify it as 'out_of_scope'.
"""

            # Call OpenRouter API
            response = self._call_openrouter_api(prompt, max_tokens=200)

            if response:
                try:
                    # Parse JSON response
                    result = json.loads(response)

                    # Validate required fields
                    if 'primary_intent' in result and 'confidence' in result:
                        primary_intent = result['primary_intent']
                        confidence = float(result['confidence'])

                        # Validate intent exists in our categories
                        if primary_intent in self.intent_categories:
                            # Apply conversation context boost
                            context_boost = self._apply_openrouter_context_boost(
                                primary_intent, message, conversation_history
                            )

                            # Final confidence with context boost
                            final_confidence = min(confidence + context_boost, 1.0)

                            return {
                                'primary_intent': primary_intent,
                                'confidence': final_confidence,
                                'all_scores': {primary_intent: final_confidence},
                                'multi_intent': result.get('multi_intent', []),
                                'classification_method': 'openrouter_ai',
                                'language_detected': result.get('language_detected', 'en'),
                                'urgency_level': result.get('urgency_level', 'low'),
                                'ai_reasoning': result.get('reasoning', '')
                            }

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse OpenRouter JSON response: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing OpenRouter response: {e}")

            return None

        except Exception as e:
            self.logger.error(f"OpenRouter AI-powered classification failed: {e}")
            return None

    def _build_conversation_context(self, conversation_history: List[Dict[str, Any]] = None) -> str:
        """Build conversation context for OpenRouter API."""
        if not conversation_history:
            return "No previous conversation"

        try:
            context_parts = []
            recent_turns = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history

            for turn in recent_turns:
                if isinstance(turn, dict):
                    intent = turn.get('intent') or turn.get('primary_intent', 'unknown')
                    content = turn.get('content', '')[:50]  # Truncate long content
                    context_parts.append(f"Previous: {intent} - '{content}'")

            return " | ".join(context_parts) if context_parts else "No previous conversation"
        except Exception:
            return "No previous conversation"

    def _apply_openrouter_context_boost(self, predicted_intent: str, current_message: str,
                                      conversation_history: List[Dict[str, Any]] = None) -> float:
        """Apply context boost based on conversation history for OpenRouter results."""
        if not conversation_history:
            return 0.0

        boost = 0.0
        try:
            # Look at recent conversation turns for context
            recent_turns = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history

            for turn in recent_turns:
                if isinstance(turn, dict):
                    # Check if previous intents are related to current prediction
                    prev_intent = turn.get('intent') or turn.get('primary_intent')
                    if prev_intent and self._are_intents_related(prev_intent, predicted_intent):
                        boost += 0.1

                    # Check for specific conversation patterns
                    prev_content = turn.get('content', '').lower()
                    current_lower = current_message.lower()

                    # If conversation is about appointments and user mentions booking -> boost
                    if any(word in prev_content for word in ['appointment', 'doctor', 'schedule']) and \
                       any(word in current_lower for word in ['book', 'schedule', 'appointment']):
                        boost += 0.15

                    # If previous message was about medicine and current is about prescription -> boost
                    if any(word in prev_content for word in ['medicine', 'tablet', 'dawai']) and \
                       any(word in current_lower for word in ['prescription', 'parchi', 'how to take']):
                        boost += 0.15

        except Exception as e:
            self.logger.warning(f"OpenRouter context boost calculation failed: {e}")

        return min(boost, 0.3)  # Cap boost at 0.3

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    def _apply_conversation_context_boost(self, predicted_intent: str, current_message: str,
                                        conversation_history: List[Dict[str, Any]] = None) -> float:
        """Apply context boost based on conversation history"""
        if not conversation_history:
            return 0.0

        boost = 0.0
        try:
            # Look at recent conversation turns for context
            recent_turns = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history

            for turn in recent_turns:
                if isinstance(turn, dict):
                    # Check if previous intents are related to current prediction
                    prev_intent = turn.get('intent') or turn.get('primary_intent')
                    if prev_intent and self._are_intents_related(prev_intent, predicted_intent):
                        boost += 0.1

                    # Check if conversation flow suggests this intent
                    prev_content = turn.get('content', '').lower()
                    current_lower = current_message.lower()

                    # If conversation is about appointments and user mentions booking -> boost
                    if any(word in prev_content for word in ['appointment', 'doctor', 'schedule']) and \
                       any(word in current_lower for word in ['book', 'schedule', 'appointment']):
                        boost += 0.15

        except Exception as e:
            self.logger.warning(f"Context boost calculation failed: {e}")

        return min(boost, 0.3)  # Cap boost at 0.3

    def validate_api_key(self) -> Dict[str, Any]:
        """Validate the OpenRouter API key and return detailed status."""
        result = {
            "api_key_configured": bool(self.api_keys),
            "validation_attempted": False,
            "validation_successful": False,
            "error_details": None,
            "recommendations": []
        }

        if not self.api_keys:
            result["error_details"] = "No API key provided"
            result["recommendations"] = [
                "Set NLU_API_KEYS environment variable in Render dashboard",
                "Visit https://openrouter.ai/keys to create an API key",
                "Ensure the API key has sufficient credits"
            ]
            return result

        result["validation_attempted"] = True

        try:
            validation_success = self._test_openrouter_connection()
            result["validation_successful"] = validation_success

            if validation_success:
                result["recommendations"] = [
                    "API key is working correctly",
                    "OpenRouter AI features are enabled",
                    "Multilingual intent recognition is available"
                ]
            else:
                result["error_details"] = "API key validation failed"
                result["recommendations"] = [
                    "Verify API key is correct and not expired",
                    "Check if API key has sufficient credits",
                    "Ensure API key has access to the specified model",
                    "Visit https://openrouter.ai/keys to manage your API key"
                ]

        except Exception as e:
            result["error_details"] = str(e)
            result["recommendations"] = [
                "Check your internet connection",
                "Verify OpenRouter service is available",
                "Try again in a few minutes",
                "Contact OpenRouter support if issue persists"
            ]

        return result

    def get_display_functions(self, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine which functions and buttons should be displayed based on intent analysis.
        Only shows functions when confidence is high enough and intent is clear.
        """
        primary_intent = intent_analysis.get('primary_intent', 'general_inquiry')
        confidence = intent_analysis.get('confidence', 0.0)
        urgency_level = intent_analysis.get('urgency_level', 'low')

        display_info = {
            'show_functions': False,
            'show_buttons': False,
            'functions': [],
            'buttons': [],
            'reasoning': 'low_confidence'
        }

        # Only show functions if confidence is above threshold
        if confidence >= self.function_display_threshold:
            display_info['show_functions'] = True
            display_info['reasoning'] = 'high_confidence'

            # Get intent-specific functions
            intent_functions = self.button_display_rules.get(primary_intent, [])

            # Add urgency-based functions for emergency situations
            if urgency_level == 'emergency':
                intent_functions.extend(['call_ambulance', 'show_emergency_contacts'])
                display_info['reasoning'] = 'emergency_situation'

            elif urgency_level == 'high':
                intent_functions.extend(['call_doctor', 'show_nearby_hospitals'])
                display_info['reasoning'] = 'high_urgency'

            display_info['functions'] = list(set(intent_functions))  # Remove duplicates

        # Show buttons for medium confidence levels
        if confidence >= 0.5 and confidence < self.function_display_threshold:
            display_info['show_buttons'] = True
            display_info['buttons'] = ['clarify_intent', 'show_suggestions']
            display_info['reasoning'] = 'medium_confidence'

        # Special handling for multilingual inputs
        language = intent_analysis.get('language_detected', 'en')
        if language in ['hi', 'pa']:
            display_info['show_language_toggle'] = True
            display_info['current_language'] = language

        return display_info

    def _are_intents_related(self, intent1: str, intent2: str) -> bool:
        """Check if two intents are related for context boosting"""
        related_groups = {
            'appointment_group': ['appointment_booking', 'appointment_view', 'appointment_cancel'],
            'medicine_group': ['find_medicine', 'medicine_scan', 'prescription_inquiry', 'prescription_upload'],
            'health_group': ['symptom_triage', 'health_record_request', 'prescription_summary_request'],
            'help_group': ['general_inquiry', 'how_to_appointment_booking', 'how_to_medicine_scan']
        }

        for group in related_groups.values():
            if intent1 in group and intent2 in group:
                return True

        return False

    def _detect_multi_intent(self, similarities: Dict[str, float], threshold: float = 0.6) -> List[str]:
        """Detect if message contains multiple intents"""
        high_confidence_intents = [
            intent for intent, score in similarities.items()
            if score >= threshold and intent != 'out_of_scope'
        ]

        return high_confidence_intents[:3]  # Return top 3 multi-intents

    def understand_user_intent(self, user_message: str, conversation_history: List[Dict[str, Any]] = None, excluded_intents: List[str] = None, sehat_sahara_mode: bool = False) -> Dict[str, Any]:
        """
        AI-powered NLU processing with semantic understanding for superior intent classification.
        Falls back to enhanced keyword-based system if AI classification fails.
        """
        cleaned_message = self._clean_and_preprocess(user_message)

        # Immediate check for out of scope content using AI if available
        if self._is_out_of_scope(cleaned_message):
            return self._generate_out_of_scope_response()
        # --- FIX: Force full analysis for first message (even if short) ---
        # This addresses your requirement 3
        if len(cleaned_message.split()) <= 4 and conversation_history:
            # Only handle short messages using history if history *exists*
            self.logger.info(f"🔄 Using short-message history-based NLU for: '{cleaned_message[:50]}...'")
            return self._handle_short_message_enhanced(cleaned_message, excluded_intents, conversation_history)
        # --- END OF FIX ---
        # (The code will now proceed to full AI/keyword analysis for short *first* messages

        # Primary: OpenRouter AI-powered classification with multilingual understanding
        if self.use_openrouter and hasattr(self, 'intent_descriptions'):
            self.logger.info(f"🔬 Using OpenRouter AI-powered NLU for message: '{cleaned_message[:50]}...'")
            ai_result = self._ai_powered_classification(cleaned_message, conversation_history)

            if ai_result and ai_result['confidence'] > 0.6:  # Confidence threshold for AI classification
                self.logger.info(f"✅ OpenRouter AI classification successful: {ai_result['primary_intent']} ({ai_result['confidence']:.2f})")
                final_result = self._compile_enhanced_final_analysis(ai_result, cleaned_message, sehat_sahara_mode)

                # Add function/button display information
                display_info = self.get_display_functions(final_result)
                final_result['display_functions'] = display_info

                return final_result

        # Secondary: Enhanced keyword-based system with better accuracy
        self.logger.info(f"🔄 Using enhanced keyword-based NLU for message: '{cleaned_message[:50]}...'")
        enhanced_result = self._get_enhanced_fallback_analysis(cleaned_message, excluded_intents, conversation_history)
        return self._compile_enhanced_final_analysis(enhanced_result, cleaned_message, sehat_sahara_mode)


    def _get_enhanced_fallback_analysis(self, message: str, excluded_intents: List[str] = None, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced fallback analysis with better keyword matching and context awareness."""

        # Enhanced short message handling with context
        if len(message.split()) <= 4:
            return self._handle_short_message_enhanced(message, excluded_intents, conversation_history)

        # Enhanced comprehensive analysis for longer messages
        analysis = self._enhanced_comprehensive_intent_detection(message, excluded_intents, conversation_history)
        urgency_analysis = self._assess_urgency_and_severity(message, analysis)
        context_entities = self._extract_health_context(message)
        language_detected = self._detect_language(message)
        user_needs = self._identify_user_needs(analysis['primary_intent'])

        return {
            'primary_intent': analysis['primary_intent'],
            'confidence': analysis['confidence'],
            'urgency_level': urgency_analysis['urgency_level'],
            'language_detected': language_detected,
            'context_entities': context_entities,
            'user_needs': user_needs,
            'in_scope': True,
            'classification_method': 'enhanced_keyword'
        }

    def _handle_short_message_enhanced(self, message: str, excluded_intents: List[str] = None, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced handling of short messages with conversation context."""

        # Check for specific intents with enhanced pattern matching
        for intent, data in self.intent_categories.items():
            if any(keyword in message for keyword in data['keywords']):
                if intent != 'general_inquiry':
                    self.logger.info(f"Short message '{message}' matched specific intent '{intent}'.")
                    analysis = self._enhanced_comprehensive_intent_detection(message, excluded_intents, conversation_history)
                    urgency_analysis = self._assess_urgency_and_severity(message, analysis)
                    context_entities = self._extract_health_context(message)
                    language_detected = self._detect_language(message)
                    user_needs = self._identify_user_needs(analysis['primary_intent'])

                    # Apply conversation context boost for short messages
                    context_boost = self._calculate_short_message_context_boost(message, conversation_history)
                    final_confidence = min(analysis['confidence'] + context_boost, 0.95)

                    return {
                        'primary_intent': analysis['primary_intent'],
                        'confidence': final_confidence,
                        'urgency_level': urgency_analysis['urgency_level'],
                        'language_detected': language_detected,
                        'context_entities': context_entities,
                        'user_needs': user_needs,
                        'in_scope': True,
                        'classification_method': 'enhanced_keyword'
                    }

        # Enhanced general inquiry detection with conversation context
        context_boost = self._calculate_short_message_context_boost(message, conversation_history)
        base_confidence = 0.7 + context_boost

        return {
            'primary_intent': 'general_inquiry',
            'confidence': min(base_confidence, 0.9),
            'urgency_level': 'low',
            'language_detected': self._detect_language(message),
            'context_entities': {},
            'user_needs': ['guidance'],
            'in_scope': True,
            'classification_method': 'enhanced_keyword'
        }

    def _calculate_short_message_context_boost(self, message: str, conversation_history: List[Dict[str, Any]] = None) -> float:
        """Calculate context boost for short messages based on conversation history."""
        if not conversation_history:
            return 0.0

        boost = 0.0
        try:
            # Look at recent conversation for context clues
            recent_turns = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history

            for turn in recent_turns:
                if isinstance(turn, dict):
                    prev_intent = turn.get('intent') or turn.get('primary_intent')
                    if prev_intent and prev_intent != 'general_inquiry':
                        # If previous conversation was about appointments and user sends "yes" -> boost appointment intent
                        if prev_intent in ['appointment_booking', 'appointment_view'] and \
                           message.lower() in ['yes', 'book', 'schedule', 'appointment']:
                            boost += 0.2

        except Exception:
            pass

        return min(boost, 0.2)

    def _enhanced_comprehensive_intent_detection(self, message: str, excluded_intents: List[str] = None, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced comprehensive intent detection with conversation context and better scoring."""
        keyword_scores = self._enhanced_keyword_intent_detection(message)

        # Apply conversation context boost to keyword scores
        if conversation_history:
            keyword_scores = self._apply_keyword_context_boost(keyword_scores, message, conversation_history)

        if excluded_intents:
            for intent in excluded_intents:
                if intent in keyword_scores:
                    keyword_scores[intent] *= 0.1

        if not keyword_scores:
            primary_intent = 'general_inquiry'
            confidence = 0.3
        else:
            primary_intent = max(keyword_scores, key=keyword_scores.get)
            confidence = keyword_scores[primary_intent]

            # Apply length-based confidence adjustment
            confidence = self._adjust_confidence_by_length(message, confidence)

        return {
            'primary_intent': primary_intent,
            'confidence': min(confidence, 1.0),
            'all_scores': keyword_scores
        }

    def _apply_keyword_context_boost(self, scores: Dict[str, float], message: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Apply context boost to keyword scores based on conversation history."""
        try:
            recent_turns = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history

            for turn in recent_turns:
                if isinstance(turn, dict):
                    prev_intent = turn.get('intent') or turn.get('primary_intent')
                    if prev_intent and prev_intent in scores:
                        # Boost related intents based on conversation flow
                        if prev_intent in ['appointment_booking', 'appointment_view']:
                            # If previous conversation was about appointments, boost appointment-related intents
                            related_intents = ['appointment_booking', 'appointment_view', 'appointment_cancel']
                            for intent in related_intents:
                                if intent in scores:
                                    scores[intent] *= 1.2

                        elif prev_intent in ['medicine_scan', 'prescription_inquiry']:
                            # If previous conversation was about medicine, boost medicine-related intents
                            related_intents = ['find_medicine', 'medicine_scan', 'prescription_inquiry', 'prescription_upload']
                            for intent in related_intents:
                                if intent in scores:
                                    scores[intent] *= 1.2

        except Exception:
            pass

        return scores

    def _adjust_confidence_by_length(self, message: str, base_confidence: float) -> float:
        """Adjust confidence based on message length and complexity."""
        word_count = len(message.split())

        if word_count <= 3:
            # Short messages - slightly lower confidence unless very clear
            return base_confidence * 0.9
        elif word_count <= 8:
            # Medium messages - standard confidence
            return base_confidence
        else:
            # Long messages - potentially higher confidence due to more context
            return min(base_confidence * 1.1, 0.95)

    def _enhanced_keyword_intent_detection(self, message: str) -> Dict[str, float]:
        """Detects intent based on keywords with multilingual support."""
        scores = {}
        for category, data in self.intent_categories.items():
            score = 0.0
            for keyword in data['keywords']:
                if keyword in message:
                    score += 0.2 * len(keyword.split())  # Weight longer phrases more

            # Boost score for urgency indicators
            for urgency_indicator in data.get('urgency_indicators', []):
                if re.search(r'\b' + re.escape(urgency_indicator) + r'\b', message, re.IGNORECASE):
                    score *= 1.5

            if score > 0:
                scores[category] = min(score, 1.0)
        return scores

    def _assess_urgency_and_severity(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Assesses urgency based on keywords and intent for health app context."""
        intent = analysis['primary_intent']
        
        # Emergency keywords
        emergency_keywords = ['emergency', 'accident', 'ambulance', 'help me', 'urgent help', 
                             'emergency hai', 'accident hua hai', 'turant help', 'emergency call']
        
        # High urgency symptoms
        urgent_symptoms = ['chest pain', 'breathing problem', 'severe pain', 'unconscious',
                          'chest mein dard', 'saans nahi aa rahi', 'behosh']
        
        urgency_level = 'low'
        
        if intent == 'emergency_assistance' or any(keyword in message.lower() for keyword in emergency_keywords):
            urgency_level = 'emergency'
        elif any(symptom in message.lower() for symptom in urgent_symptoms):
            urgency_level = 'high'
        elif intent == 'symptom_triage':
            urgency_level = 'medium'

        return {
            'urgency_level': urgency_level
        }

    def _extract_health_context(self, message: str) -> Dict[str, str]:
        """Extracts health-related entities from the message."""
        context = {}
        
        # Doctor specialties
        specialties = ['cardiologist', 'dermatologist', 'pediatrician', 'gynecologist', 
                      'orthopedic', 'neurologist', 'heart doctor', 'skin doctor', 'child doctor']
        for specialty in specialties:
            if specialty in message.lower():
                context['doctor_type'] = specialty
                break
        
        # Common symptoms
        symptoms = ['fever', 'headache', 'cough', 'pain', 'cold', 'bukhar', 'sir dard', 'khansi']
        for symptom in symptoms:
            if symptom in message.lower():
                context['symptom'] = symptom
                break
        
        return context

    def _detect_language(self, message: str) -> str:
        """Improved language detection with better accuracy for mixed content."""
        if not message or not message.strip():
            return 'en'

        message = message.strip()
        message_lower = message.lower()

        # Script-based detection (highest priority)
        script_hindi = bool(re.search(r'[\u0900-\u097F]', message))  # Devanagari script
        script_punjabi = bool(re.search(r'[\u0A00-\u0A7F]', message))  # Gurmukhi script

        if script_hindi:
            return 'hi'
        elif script_punjabi:
            return 'pa'

        word_count = len(message.split())
        if word_count <= 3:
            # Check for definitive language markers
            hindi_markers = ['hai', 'kya', 'nahi', 'chahiye', 'karna', 'mera', 'meri', 'tera', 'teri']
            punjabi_markers = ['hai', 'ki', 'nahin', 'chahidi', 'karna', 'mera', 'meri', 'tera', 'teri', 'kado', 'kithe', 'kive']
            english_markers = ['the', 'is', 'are', 'do', 'have', 'my', 'i', 'you', 'this', 'that', 'what', 'how', 'when', 'where', 'why']

            # Count exact word matches
            hi_count = sum(1 for word in message_lower.split() if word in hindi_markers)
            pa_count = sum(1 for word in message_lower.split() if word in punjabi_markers)
            en_count = sum(1 for word in message_lower.split() if word in english_markers)

            if hi_count > pa_count and hi_count > en_count:
                return 'hi'
            elif pa_count > hi_count and pa_count > en_count:
                return 'pa'
            elif en_count > 0:
                return 'en'
            else:
                return 'en'  # Default for ambiguous short messages

        keyword_scores = {'en': 0, 'hi': 0, 'pa': 0}

        # English keywords with weights
        english_words = {
            'the': 2, 'is': 2, 'are': 2, 'do': 2, 'have': 2, 'my': 2, 'i': 2, 'you': 2,
            'this': 1, 'that': 1, 'what': 1, 'how': 1, 'when': 1, 'where': 1, 'why': 1,
            'fever': 1, 'headache': 1, 'pain': 1, 'doctor': 1, 'appointment': 1
        }
        for word, weight in english_words.items():
            if word in message_lower.split():
                keyword_scores['en'] += weight

        # Hindi keywords with weights (Latin script)
        hindi_words = {
            'hai': 3, 'kya': 2, 'kaise': 2, 'kab': 2, 'kahan': 2, 'meri': 2, 'mera': 2,
            'teri': 2, 'tera': 2, 'chahiye': 2, 'leni': 1, 'karna': 1, 'karne': 1,
            'nahi': 2, 'bhi': 1, 'par': 1, 'aur': 1, 'bukhar': 1, 'dard': 1, 'khansi': 1
        }
        for word, weight in hindi_words.items():
            if word in message_lower.split():
                keyword_scores['hi'] += weight

        # Punjabi keywords with weights (Latin script)
        punjabi_words = {
            'hai': 3, 'ki': 2, 'kive': 3, 'kado': 3, 'kithe': 3, 'meri': 2, 'mera': 2,
            'teri': 2, 'tera': 2, 'chahidi': 3, 'leni': 1, 'karna': 1, 'karne': 1,
            'nahin': 2, 'bhi': 1, 'par': 1, 'aur': 1, 'bukhar': 1, 'dukh': 2, 'khansi': 1
        }
        for word, weight in punjabi_words.items():
            if word in message_lower.split():
                keyword_scores['pa'] += weight

        # Find language with highest score
        best_language = max(keyword_scores, key=keyword_scores.get)
        max_score = keyword_scores[best_language]

        # Only return result if we have reasonable confidence
        if max_score > 0:
            return best_language
        else:
            return None  # Default fallback

    def _identify_user_needs(self, primary_intent: str) -> List[str]:
        """Identifies user needs based on intent."""
        need_mapping = {
            'appointment_booking': ['booking', 'doctor_connection'],
            'appointment_view': ['information', 'schedule_check'],
            'appointment_cancel': ['booking_management'],
            'health_record_request': ['information', 'record_access'],
            'symptom_triage': ['health_assessment', 'guidance'],
            'find_medicine': ['pharmacy_search', 'medicine_availability'],
            'prescription_inquiry': ['information', 'medicine_guidance'],
            'prescription_upload': ['prescription_management', 'document_upload'],
            'medicine_scan': ['medicine_identification'],
            'emergency_assistance': ['immediate_help', 'emergency_services'],
            'report_issue': ['feedback', 'complaint_handling'],
            'general_inquiry': ['guidance', 'app_navigation'],
            'out_of_scope': ['redirection'],
            'set_medicine_reminder': ['reminder_management', 'medicine_connection']
        }
        return need_mapping.get(primary_intent, ['guidance'])

    def _is_out_of_scope(self, message: str) -> bool:
        """Check if message is out of scope for health app."""
        out_of_scope_keywords = self.intent_categories['out_of_scope']['keywords']
        return any(keyword in message.lower() for keyword in out_of_scope_keywords)

    def _generate_out_of_scope_response(self) -> Dict[str, Any]:
        """Returns structured response for out of scope content."""
        return {
            'primary_intent': 'out_of_scope',
            'confidence': 0.95,
            'urgency_level': 'low',
            'language_detected': 'en',
            'context_entities': {},
            'user_needs': ['redirection'],
            'in_scope': False
        }

    def _clean_and_preprocess(self, message: str) -> str:
        """Cleans and standardizes the user's message for analysis."""
        cleaned = message.lower().strip()
        contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not", "didn't": "did not",
            "i'm": "i am", "you're": "you are", "it's": "it is", "i've": "i have"
        }

        for contraction, expansion in contractions.items():
            cleaned = cleaned.replace(contraction, expansion)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        return cleaned

    def _compile_enhanced_final_analysis(self, analysis_data: Dict[str, Any], cleaned_message: str, sehat_sahara_mode: bool = False) -> Dict[str, Any]:
        """Enhanced compilation of final NLU response with AI-powered insights."""
        primary_intent_value = analysis_data.get('primary_intent', 'general_inquiry')
        if isinstance(primary_intent_value, list) and len(primary_intent_value) > 0:
            primary_intent_value = primary_intent_value[0]
        elif not isinstance(primary_intent_value, str):
            primary_intent_value = 'general_inquiry'

        # Enhanced conversation stage determination with AI context
        conversation_stage = self._determine_enhanced_conversation_stage(
            cleaned_message,
            {'primary_intent': primary_intent_value, 'confidence': analysis_data.get('confidence', 0.5)},
            analysis_data
        )

        # Enhanced language detection with AI confidence
        language_detected = analysis_data.get('language_detected', 'en')
        if sehat_sahara_mode and language_detected not in ['hi', 'pa', 'en']:
            language_detected = self._detect_language(cleaned_message)

        # Enhanced context entities extraction
        context_entities = analysis_data.get('context_entities', {})
        if analysis_data.get('classification_method') == 'ai_semantic':
            # AI can extract more sophisticated context entities
            context_entities = self._extract_enhanced_context_entities(cleaned_message, primary_intent_value)

        # Determine if AI was used for classification
        ai_analysis_used = analysis_data.get('classification_method') == 'ai_semantic'

        result = {
            'primary_intent': primary_intent_value,
            'confidence': float(analysis_data.get('confidence', 0.5)),
            'urgency_level': analysis_data.get('urgency_level', 'low'),
            'language_detected': language_detected,
            'context_entities': context_entities,
            'conversation_stage': conversation_stage,
            'user_needs': analysis_data.get('user_needs', ['guidance']),
            'in_scope': bool(analysis_data.get('in_scope', True)),
            'processing_timestamp': datetime.now().isoformat(),
            'api_analysis_used': ai_analysis_used,
            'classification_method': analysis_data.get('classification_method', 'enhanced_keyword'),
            'multi_intent': analysis_data.get('multi_intent', []),
            'all_scores': analysis_data.get('all_scores', {})
        }

        return result

    def _determine_enhanced_conversation_stage(self, message: str, analysis: Dict, full_analysis: Dict) -> str:
        """Enhanced conversation stage determination with AI-powered context."""
        intent = analysis['primary_intent']
        confidence = analysis.get('confidence', 0.5)

        # High confidence AI classification can lead to more specific stages
        if full_analysis.get('classification_method') == 'ai_semantic' and confidence > 0.8:
            # AI with high confidence can determine more nuanced stages
            if intent == 'appointment_booking' and any(word in message.lower() for word in ['urgent', 'emergency', 'asap']):
                return 'urgent_task_execution'
            elif intent in ['medicine_scan', 'prescription_upload'] and confidence > 0.85:
                return 'document_processing'

        # Standard stage determination
        if intent == 'emergency_assistance':
            return 'emergency_handling'
        elif intent in ['appointment_booking', 'find_medicine', 'medicine_scan', 'set_medicine_reminder']:
            return 'task_execution'
        elif intent in ['appointment_view', 'health_record_request', 'prescription_inquiry']:
            return 'information_retrieval'
        else:
            return 'understanding'

    def _extract_enhanced_context_entities(self, message: str, primary_intent: str) -> Dict[str, str]:
        """Extract enhanced context entities using AI-powered understanding."""
        context = {}

        # Enhanced entity extraction based on intent
        if primary_intent == 'appointment_booking':
            # Extract time preferences
            time_patterns = ['tomorrow', 'today', 'morning', 'afternoon', 'evening', 'urgent', 'asap']
            for pattern in time_patterns:
                if pattern in message.lower():
                    context['time_preference'] = pattern

            # Extract doctor preferences
            specialties = ['cardiologist', 'dermatologist', 'pediatrician', 'gynecologist', 'orthopedic']
            for specialty in specialties:
                if specialty in message.lower():
                    context['doctor_type'] = specialty

        elif primary_intent == 'symptom_triage':
            # Extract symptom severity
            severity_indicators = ['severe', 'mild', 'moderate', 'extreme', 'terrible']
            for indicator in severity_indicators:
                if indicator in message.lower():
                    context['severity'] = indicator

            # Extract duration
            duration_patterns = ['days', 'weeks', 'hours', 'chronic', 'sudden']
            for pattern in duration_patterns:
                if pattern in message.lower():
                    context['duration'] = pattern

        elif primary_intent in ['medicine_scan', 'prescription_inquiry']:
            # Extract medicine-related context
            if any(word in message.lower() for word in ['tablet', 'capsule', 'syrup', 'injection']):
                context['medicine_form'] = 'specified'

        return context

    def _determine_conversation_stage(self, message: str, analysis: Dict) -> str:
        """Determines the current stage of the conversation for health app context."""
        intent = analysis['primary_intent']
        
        if intent == 'emergency_assistance':
            return 'emergency_handling'
        elif intent in ['appointment_booking', 'find_medicine', 'medicine_scan', 'set_medicine_reminder']:
            return 'task_execution'
        elif intent in ['appointment_view', 'health_record_request', 'prescription_inquiry']:
            return 'information_retrieval'
        else:
            return 'understanding'

    # Backward compatibility and utility methods
    def save_nlu_model(self, filepath: str) -> bool:
        """Save NLU model configuration and learned parameters."""
        try:
            with self._lock:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                config = {
                    'intent_categories': self.intent_categories,
                    'conversation_stages': self.conversation_stages,
                    'use_semantic': self.use_semantic,
                    'model_version': '3.0.0',
                    'save_timestamp': datetime.now().isoformat()
                }
                
                if hasattr(self, 'category_embeddings') and self.category_embeddings:
                    config['category_embeddings'] = {
                        category: embedding.tolist() 
                        for category, embedding in self.category_embeddings.items()
                    }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(config, f)
                
                self.logger.info(f"✅ NLU model configuration saved to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save NLU model: {e}")
            return False

    def load_nlu_model(self, filepath: str) -> bool:
        """Load NLU model configuration and parameters."""
        try:
            with self._lock:
                with open(filepath, 'rb') as f:
                    config = pickle.load(f)
                
                self.intent_categories = config.get('intent_categories', self.intent_categories)
                self.conversation_stages = config.get('conversation_stages', self.conversation_stages)
                self.ollama_model = config.get('ollama_model', self.ollama_model)
                
                if 'category_embeddings' in config and self.use_semantic:
                    self.category_embeddings = {
                        category: np.array(embedding) 
                        for category, embedding in config['category_embeddings'].items()
                    }
                
                self.logger.info(f"✅ NLU model configuration loaded from {filepath}")
                return True
                
        except FileNotFoundError:
            self.logger.warning(f"⚠️ NLU model file not found: {filepath}. Using defaults.")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading NLU model: {e}. Using defaults.")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current OpenRouter AI-enhanced model configuration."""
        return {
            'model_type': 'Sehat Sahara Health Assistant OpenRouter AI-Powered NLU Processor',
            'version': '5.0.0',
            'ai_powered': True,
            'classification_method': 'OpenRouter AI + Enhanced Keyword Fallback',
            'openrouter_enabled': self.use_openrouter,
            'openrouter_model': self.openrouter_model if self.use_openrouter else None,
            'legacy_semantic_enabled': self.use_semantic,
            'sentence_transformer_model': 'paraphrase-multilingual-MiniLM-L12-v2' if self.use_semantic else None,
            'intent_categories_count': len(self.intent_categories),
            'conversation_stages_count': len(self.conversation_stages),
            'supported_languages': ['English', 'Hindi', 'Punjabi', 'Multilingual'],
            'ai_confidence_threshold': 0.6,
            'function_display_threshold': self.function_display_threshold,
            'enhanced_features': [
                'OpenRouter AI-powered multilingual classification',
                'Intelligent function/button display management',
                'Advanced conversation context awareness',
                'Multi-intent detection with confidence scoring',
                'Enhanced entity extraction for health context',
                'Intelligent fallback system with graceful degradation',
                'Context-based confidence boosting',
                'Urgency-based function prioritization',
                'Multilingual support with language detection'
            ],
            'initialized_at': datetime.now().isoformat()
        }

    def test_ai_nlu_transformation(self) -> Dict[str, Any]:
        """Test the transformed AI-powered NLU with various inputs to demonstrate improvements."""
        test_cases = [
            # Test cases that would fail with keyword-only approach but work with AI
            {
                'input': "I need medical consultation for my health issues",
                'expected_traditional': 'general_inquiry',  # Would fail with keyword approach
                'expected_ai': 'appointment_booking'  # Should work with AI semantic understanding
            },
            {
                'input': "Can you help me identify what medicine this is from the image?",
                'expected_traditional': 'general_inquiry',  # Would fail with keyword approach
                'expected_ai': 'medicine_scan'  # Should work with AI semantic understanding
            },
            {
                'input': "I'm experiencing severe chest pain and need urgent help",
                'expected_traditional': 'symptom_triage',  # Basic keyword match
                'expected_ai': 'emergency_assistance'  # AI should recognize urgency better
            },
            {
                'input': "How should I take these tablets that the doctor prescribed?",
                'expected_traditional': 'general_inquiry',  # Would fail with keyword approach
                'expected_ai': 'prescription_inquiry'  # Should work with AI semantic understanding
            },
            # Multilingual test cases
            {
                'input': "मुझे अपनी दवाई की जानकारी चाहिए",  # Hindi: "I need information about my medicine"
                'expected_traditional': 'general_inquiry',  # Would fail with keyword approach
                'expected_ai': 'prescription_inquiry'  # Should work with AI multilingual understanding
            },
            {
                'input': "ਮੈਨੂੰ ਡਾਕਟਰ ਨਾਲ ਮਿਲਣਾ ਹੈ",  # Punjabi: "I need to meet a doctor"
                'expected_traditional': 'general_inquiry',  # Would fail with keyword approach
                'expected_ai': 'appointment_booking'  # Should work with AI multilingual understanding
            }
        ]

        results = {
            'total_tests': len(test_cases),
            'ai_successful': 0,
            'traditional_successful': 0,
            'ai_results': [],
            'improvement_demonstrated': False
        }

        for i, test_case in enumerate(test_cases):
            test_input = test_case['input']

            # Test with AI classification
            ai_result = self.understand_user_intent(test_input)
            ai_intent = ai_result.get('primary_intent', 'general_inquiry')
            ai_confidence = ai_result.get('confidence', 0.0)
            ai_method = ai_result.get('classification_method', 'unknown')

            # Traditional keyword-only would have different results for complex cases
            traditional_intent = test_case['expected_traditional']
            ai_expected = test_case['expected_ai']

            # Check if AI performed better
            ai_improved = (ai_intent == ai_expected and traditional_intent != ai_expected)

            if ai_intent == ai_expected:
                results['ai_successful'] += 1

            if ai_intent == traditional_intent:
                results['traditional_successful'] += 1

            if ai_improved:
                results['improvement_demonstrated'] = True

            results['ai_results'].append({
                'test_case': i + 1,
                'input': test_input,
                'ai_intent': ai_intent,
                'ai_confidence': round(ai_confidence, 3),
                'ai_method': ai_method,
                'traditional_expected': traditional_intent,
                'ai_expected': ai_expected,
                'ai_improved': ai_improved
            })

        results['ai_accuracy'] = results['ai_successful'] / results['total_tests']
        results['traditional_accuracy'] = results['traditional_successful'] / results['total_tests']
        results['improvement_ratio'] = results['ai_accuracy'] / max(results['traditional_accuracy'], 0.1)

        return results

    def validate_configuration(self) -> bool:
        """Validate the current model configuration."""
        try:
            if not self.intent_categories:
                self.logger.error("❌ No intent categories defined")
                return False
            
            if not self.conversation_stages:
                self.logger.error("❌ No conversation stages defined")
                return False
            
            test_result = self.understand_user_intent("book appointment")
            if not test_result or 'primary_intent' not in test_result:
                self.logger.error("❌ Basic intent detection failed")
                return False
            
            self.logger.info("✅ NLU model configuration is valid")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False

    # Backward compatibility methods
    def analyze_user_message(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Backward compatibility method - alias for understand_user_intent."""
        return self.understand_user_intent(message, context.get('excluded_intents') if context else None)

    def get_intent_confidence(self, message: str, intent: str) -> float:
        """Get confidence score for a specific intent."""
        result = self.understand_user_intent(message)
        return result.get('confidence', 0.0) if result.get('primary_intent') == intent else 0.0

    def is_emergency_detected(self, message: str) -> bool:
        """Quick check if message indicates emergency situation."""
        result = self.understand_user_intent(message)
        return result['urgency_level'] == 'emergency' or result['primary_intent'] == 'emergency_assistance'