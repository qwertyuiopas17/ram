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

# Try to import advanced NLP libraries with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class ProgressiveNLUProcessor:
    """
    NLU processor for Sehat Sahara Health Assistant with multilingual support.
    Processes user commands for health app navigation and task completion.
    """

    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        # Initialize semantic model for enhanced understanding (optional)
        self.sentence_model = None
        self.use_semantic = False
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.use_semantic = True
                self.logger.info("✅ Semantic model loaded for enhanced NLU")
            except Exception as e:
                self.logger.warning(f"Could not load semantic model: {e}")

        # Health app intent categories with multilingual keywords
        self.intent_categories = {
            'appointment_booking': {
                'keywords': [
                    # English
                    'book appointment', 'need to see doctor', 'doctor appointment', 'schedule appointment',
                    'meet doctor', 'consultation', 'book doctor', 'see doctor', 'doctor visit',
                    # Hindi (Latin script)
                    'doctor se milna hai', 'appointment book karni hai', 'doctor ko dikhana hai',
                    'doctor ke paas jana hai', 'appointment chahiye', 'doctor se baat karni hai',
                    # Punjabi (Latin script)
                    'doctor nu milna hai', 'appointment book karni hai', 'doctor kol jana hai',
                    'doctor nu dikhana hai', 'doctor de kol appointment', 'vaid nu milna hai'
                ],
                'urgency_indicators': ['urgent', 'emergency', 'turant', 'jaldi', 'emergency hai']
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
                    'tablet kitni', 'medicine timing', 'prescription details',
                    # Hindi (Latin script)
                    'dawai kaise leni hai', 'kitni tablet leni hai', 'dawai ka time',
                    'medicine kab leni hai', 'dawai ki jankari',
                    # Punjabi (Latin script)
                    'dawai kive leni hai', 'kinni tablet leni hai', 'dawai da time',
                    'medicine kado leni hai', 'dawai di jankari'
                ]
            },
            'medicine_scan': {
                'keywords': [
                    # English
                    'scan medicine', 'check medicine', 'medicine scanner', 'identify medicine',
                    'what is this medicine', 'medicine name',
                    # Hindi (Latin script)
                    'medicine scan karo', 'ye kya dawai hai', 'medicine check karo',
                    'dawai ka naam', 'medicine identify karo',
                    # Punjabi (Latin script)
                    'medicine scan karo', 'eh ki dawai hai', 'medicine check karo',
                    'dawai da naam', 'medicine identify karo'
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

        # Build semantic embeddings if available
        if self.use_semantic:
            self._build_semantic_embeddings()

        # Load saved model if available
        if model_path and os.path.exists(model_path):
            self.load_nlu_model(model_path)

    def _build_semantic_embeddings(self):
        """Build semantic embeddings for each intent category"""
        try:
            self.category_embeddings = {}
            for category, data in self.intent_categories.items():
                # Use keywords to create pseudo-sentences for embedding
                keywords = data['keywords'][:5]  # Use top 5 keywords
                pseudo_sentences = [f"I want to {keyword}" for keyword in keywords]
                
                # Create embeddings
                embeddings = self.sentence_model.encode(pseudo_sentences)
                
                # Use mean embedding as category representation
                self.category_embeddings[category] = np.mean(embeddings, axis=0)
            
            self.logger.info("✅ Semantic embeddings built for all intent categories")
        except Exception as e:
            self.logger.error(f"Failed to build semantic embeddings: {e}")
            self.use_semantic = False

    def understand_user_intent(self, user_message: str, conversation_history: List[Dict[str, Any]] = None, excluded_intents: List[str] = None, sehat_sahara_mode: bool = False) -> Dict[str, Any]:
        """
        Processes a user's message to understand intent and urgency for health app navigation.
        """
        cleaned_message = self._clean_and_preprocess(user_message)
        
        # Immediate check for out of scope content
        if self._is_out_of_scope(cleaned_message):
            return self._generate_out_of_scope_response()

        # Use keyword-based analysis only (removed API-based analysis)

        # Fallback to keyword-based system
        self.logger.info(f"Using keyword-based NLU for message: '{cleaned_message[:50]}...'")
        fallback_result = self._get_fallback_analysis(cleaned_message, excluded_intents)
        return self._compile_final_analysis(fallback_result, cleaned_message, sehat_sahara_mode)


    def _get_fallback_analysis(self, message: str, excluded_intents: List[str] = None) -> Dict[str, Any]:
        """Generates NLU analysis using keywords, with improved logic for short messages."""

        # First, check for specific intents even in short messages
        for intent, data in self.intent_categories.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', message, re.IGNORECASE) for keyword in data['keywords']):
                if intent != 'general_inquiry':
                    # If a specific keyword is found, immediately classify with that intent
                    self.logger.info(f"Short message '{message}' matched specific intent '{intent}'.")
                    analysis = self._comprehensive_intent_detection(message, excluded_intents)
                    urgency_analysis = self._assess_urgency_and_severity(message, analysis)
                    context_entities = self._extract_health_context(message)
                    language_detected = self._detect_language(message)
                    user_needs = self._identify_user_needs(analysis['primary_intent'])

                    # Boost confidence for clear short commands
                    analysis['confidence'] = 0.95

                    return {
                        'primary_intent': analysis['primary_intent'],
                        'confidence': analysis['confidence'],
                        'urgency_level': urgency_analysis['urgency_level'],
                        'language_detected': language_detected,
                        'context_entities': context_entities,
                        'user_needs': user_needs,
                        'in_scope': True
                    }

        # If no specific keywords are found in a short message, THEN it's a general inquiry
        if len(message.split()) <= 4: # Increased threshold to catch more conversational phrases
            self.logger.info(f"Short message without specific keywords: '{message}'. Using general_inquiry.")
            return {
                'primary_intent': 'general_inquiry',
                'confidence': 0.7,
                'urgency_level': 'low',
                'language_detected': self._detect_language(message),
                'context_entities': {},
                'user_needs': ['guidance'],
                'in_scope': True
            }

        # Perform standard analysis for longer messages
        analysis = self._comprehensive_intent_detection(message, excluded_intents)
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
            'in_scope': True
        }

    def _comprehensive_intent_detection(self, message: str, excluded_intents: List[str] = None) -> Dict[str, Any]:
        """Combines keyword matching for health app intent detection."""
        keyword_scores = self._enhanced_keyword_intent_detection(message)
        
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

        return {
            'primary_intent': primary_intent,
            'confidence': min(confidence, 1.0),
            'all_scores': keyword_scores
        }

    def _enhanced_keyword_intent_detection(self, message: str) -> Dict[str, float]:
        """Detects intent based on keywords with multilingual support."""
        scores = {}
        for category, data in self.intent_categories.items():
            score = 0.0
            for keyword in data['keywords']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', message, re.IGNORECASE):
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
            return 'en'  # Default fallback

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

    def _compile_final_analysis(self, analysis_data: Dict[str, Any], cleaned_message: str, sehat_sahara_mode: bool = False) -> Dict[str, Any]:
        """Compiles the final NLU response object from the analysis data."""
        primary_intent_value = analysis_data.get('primary_intent', 'general_inquiry')
        if isinstance(primary_intent_value, list) and len(primary_intent_value) > 0:
            primary_intent_value = primary_intent_value[0]
        elif not isinstance(primary_intent_value, str):
            primary_intent_value = 'general_inquiry'

        conversation_stage = self._determine_conversation_stage(cleaned_message, {'primary_intent': primary_intent_value})

        # For Sehat Sahara strict mode, ensure language detection is more reliable
        language_detected = analysis_data.get('language_detected', 'en')
        if sehat_sahara_mode and language_detected not in ['hi', 'pa', 'en']:
            language_detected = self._detect_language(cleaned_message)

        result = {
            'primary_intent': primary_intent_value,
            'confidence': float(analysis_data.get('confidence', 0.5)),
            'urgency_level': analysis_data.get('urgency_level', 'low'),
            'language_detected': language_detected,
            'context_entities': analysis_data.get('context_entities', {}),
            'conversation_stage': conversation_stage,
            'user_needs': analysis_data.get('user_needs', ['guidance']),
            'in_scope': bool(analysis_data.get('in_scope', True)),
            'processing_timestamp': datetime.now().isoformat(),
            'api_analysis_used': False
        }

        return result

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
        """Get information about the current model configuration."""
        return {
            'model_type': 'Sehat Sahara Health Assistant NLU Processor',
            'version': '3.0.0',
            'api_enabled': False,
            'api_model': None,
            'semantic_enabled': self.use_semantic,
            'intent_categories_count': len(self.intent_categories),
            'conversation_stages_count': len(self.conversation_stages),
            'supported_languages': ['English', 'Hindi', 'Punjabi'],
            'initialized_at': datetime.now().isoformat()
        }

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
