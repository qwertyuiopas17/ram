# API-Based Ollama Integration for Mental Health Chatbot - COMPLETE VERSION
# Modified to use API key-based service (like Groq) instead of local Ollama
# Provides seamless integration with external API while maintaining same interface

# In api_ollama_integration.py
import json
import logging
import requests
import os
import time # Import the time module
from typing import Dict, Any, List, Optional
import re 
class ApiClient:
    """Enhanced client for interacting with API-based LLM service with rotating API keys."""
    
    def __init__(self, api_keys: List[str] = None, base_url: str = "https://api.groq.com/openai/v1", model: str = "llama-3.1-8b-instant"):
        # --- MODIFICATION START ---
        self.api_keys = api_keys or [key.strip() for key in os.getenv('GROQ_API_KEYS', '').split(',') if key.strip()]
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        self.current_key_index = 0
        self.last_switch_time = 0
        # --- MODIFICATION END ---
        
        self.is_available = self.check_availability()
        
    @property
    def api_key(self):
        """Returns the currently active API key."""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def switch_key(self):
        """Rotates to the next API key in the list."""
        # Add a cooldown to prevent rapid switching if all keys are failing
        if time.time() - self.last_switch_time < 10: # 10-second cooldown
            self.logger.warning("Key switch attempted too quickly. Waiting.")
            time.sleep(10)

        if len(self.api_keys) > 1:
            old_key_preview = f"...{self.api_key[-4:]}" if self.api_key else "None"
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            new_key_preview = f"...{self.api_key[-4:]}" if self.api_key else "None"
            self.logger.warning(f"Switching API key from {old_key_preview} to {new_key_preview}")
            self.last_switch_time = time.time()
            return True
        self.logger.warning("No alternative API keys available to switch to.")
        return False

    def check_availability(self) -> bool:
        """Check if any API key is valid."""
        if not self.api_keys:
            self.logger.warning("No API keys provided. Set GROQ_API_KEYS environment variable.")
            return False
        
        # Test with the current key
        if not self.api_key:
            return False

        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            test_payload = {"model": self.model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"API service available with model {self.model}")
                return True
            else:
                self.logger.warning(f"API test failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"API service not available: {e}")
            return False
    
    def _make_request(self, payload: Dict[str, Any], retries: int = 1) -> Optional[str]:
        """Internal method to make requests, with key rotation on rate limit errors."""
        if not self.is_available:
            return None

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=90)

            # --- RATE LIMIT HANDLING ---
            if response.status_code == 429: # Too Many Requests
                self.logger.warning(f"Rate limit exceeded for key ...{self.api_key[-4:]}. Attempting to switch key.")
                if self.switch_key() and retries > 0:
                    self.logger.info("Retrying request with new key...")
                    return self._make_request(payload, retries=retries - 1)
                else:
                    self.logger.error("All API keys are rate-limited or no other keys available.")
                    return None

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return generated_text or None
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in API generation: {e}")
            return None

    def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate completion using API with key rotation."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        return self._make_request(payload)

    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
        """Generate chat completion with key rotation."""
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        return self._make_request(payload)

    # test_connection method remains the same
    def test_connection(self) -> Dict[str, Any]:
        """Test the API connection and return status"""
        test_result = {
            "available": False,
            "model": self.model,
            "base_url": self.base_url,
            "error": None,
            "test_response": None
        }
        
        try:
            if not self.is_available:
                test_result["error"] = "Service not available"
                return test_result
            
            test_prompt = "Hello, please respond with a brief greeting."
            test_response = self.generate_response(test_prompt, max_tokens=50)
            
            if test_response:
                test_result["available"] = True
                test_result["test_response"] = test_response
                self.logger.info("API connection test successful")
            else:
                test_result["error"] = "No response generated"
                self.logger.warning("API connection test failed - no response")
                
        except Exception as e:
            test_result["error"] = str(e)
            self.logger.error(f"API connection test error: {e}")
            
        return test_result

class GroqScoutClient:
    """Client for openrouter's Llama 4 Scout (used for emoji/image interpretation)."""
    def __init__(self, api_key: str = None, base_url: str = "https://openrouter.ai/api/v1", model: str = "qwen/qwen2.5-vl-32b-instruct:free"):
        # Separate API key to allow different security policy if desired
        self.api_key = api_key or os.getenv('GROQ_SCOUT_API_KEY') or os.getenv('GROQ_API_KEY') or os.getenv('API_KEY')
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.is_available = bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 90) -> Optional[Dict[str, Any]]:
        try:
            response = requests.post(f"{self.base_url}{path}", headers=self._headers(), json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            self.logger.error(f"Groq Scout API error: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Groq Scout request failed: {e}")
            return None

    def interpret_emojis(self, user_message: str, language: str = "en", context_history: List[Dict[str,str]] = None) -> Optional[str]:
        system_prompt = (
            "ROLE: You are a compassionate mental health support assistant speaking like the user's closest friend. "
            "CONTEXT: The user pasted emojis or emoji-heavy text they received and wants quick, supportive advice. "
            "GOAL: Respond like a trusted friend would—calm, caring, and practical—while staying within mental-health-safe boundaries (no diagnosis or clinical claims). "
            "INTERPRETATION: Help them read social tone. If it looks like a meme/joke/teasing, say so plainly. If intent is unclear, lean toward neutral/kind interpretation and de-escalate. "
            "If the sender seems unknown, an old friend, or an ex, note that and lean toward low-stakes, self-respecting choices. If it doesn’t look like a meme, treat it as a normal message and respond accordingly. "
            "MIRROR STYLE: Match the user's latest message script and style. If they use Hindi written in Latin letters (Hinglish), respond entirely in Hinglish. Do not switch to Devanagari. Do not include English translations in parentheses. Do not provide side-by-side translations. "
            "RESPONSE RULES: "
            "1) One short reassurance/normalization in a friendly voice. "
            "2) Offer 2–3 quick reply options based on context: one light/positive (e.g., haha/lol/thanks!), one gentle boundary (e.g., I’m not comfy with this), and if sender is unknown/ex or user seems unsure, include ignore/no-reply or ask-for-clarity. "
            "3) Ask ONE brief check-in question (e.g., is this from someone you know well, an acquaintance, or an ex?). "
            "TONE: Warm, concise, friend-to-friend; empower choice; prioritize safety and respect. "
            "STYLE: No headings, no section labels, no markdown, no long lists; under ~120 words. "
            "LANGUAGE PRIORITY: First mirror the user's style exactly (e.g., Hinglish). Otherwise write entirely in language code: " + language + ". No translations or mixing."
        )
        messages = [{"role": "system", "content": system_prompt}]
        if context_history:
            for msg in context_history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": user_message})
        payload = {"model": self.model, "messages": messages, "max_tokens": 220, "temperature": 0.55}
        result = self._post("/chat/completions", payload)
        if result:
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return None

    def interpret_medicine_image(self, user_message: str, image_b64: str, language: str = "en", context_history: List[Dict[str,str]] = None) -> Optional[str]:
        system_prompt = (
            "ROLE: You are Sehat Sahara's medicine scan helper. The user shared a photo of medicine packaging.\n"
            "GOAL: Try to identify the medicine name and provide general, non-medical info with strong safety disclaimers.\n"
            "SAFETY: NEVER provide treatment, dosage, or medical advice. Encourage consulting a doctor/pharmacist.\n"
            "OUTPUT: Under 100 words in the user's language. No markdown.\n"
            "If uncertain, say so clearly.\n"
        )
        messages = [{"role": "system", "content": system_prompt}]
        if context_history:
            for msg in context_history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        user_content = [
            {"type": "text", "text": user_message or "Please help identify this medicine from the image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}" }}
        ]
        payload = {"model": self.model, "messages": [*messages, {"role": "user", "content": user_content}], "max_tokens": 220, "temperature": 0.4}
        result = self._post("/chat/completions", payload, timeout=120)
        if result:
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return None

    def interpret_image(self, user_message: str, image_b64: str, language: str = "en", context_history: List[Dict[str,str]] = None) -> Optional[str]:
        return self.interpret_medicine_image(user_message, image_b64, language, context_history)

    # In api_ollama_integration.py, inside the GroqScoutClient class

    def interpret_prescription_image(self, image_b64: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Interpret prescription image and extract structured data"""
    # A more forceful prompt to get JSON
        system_prompt = (
            "You are a prescription analysis AI. Your ONLY job is to analyze the image and respond with a single, valid JSON object. "
            "Do not add any text, explanation, or markdown before or after the JSON. Your entire output must be the JSON itself."
            'The JSON format MUST be: {"doctor_name": "...", "medications": [{"name": "...", "dosage": "...", "time": "..."}], "tests": ["..."], "diagnosis": "..."}'
            "\nIf any information is unreadable, use empty strings or empty arrays for the corresponding fields."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this prescription and return only the JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
        ]
        payload = {"model": self.model, "messages": messages, "max_tokens": 500, "temperature": 0.1}
        result = self._post("/chat/completions", payload, timeout=120)
        if not result:
            self.logger.warning("AI did not return any result for prescription analysis.")
            return None

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # --- FIX: ADD ROBUST JSON EXTRACTION ---
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            self.logger.warning(f"No JSON object found in AI response. Response was: {content}")
            return None
    
        json_str = json_match.group(0)
        try:
        # Successfully extracted and parsed the JSON
            parsed_json = json.loads(json_str)
            self.logger.info("Successfully parsed JSON from prescription image analysis.")
            return parsed_json
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode extracted JSON from AI response. String was: {json_str}. Error: {e}")
            return None
    # --- END OF FIX ---

class SehatSaharaApiClient:
    """Enhanced mental health specific client using API service"""

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str = None, base_url: str = "https://api.groq.com/openai/v1"):
        # --- THIS IS THE FIX ---
        # Pass the api_key as a list to the 'api_keys' argument
        self.client = ApiClient(api_keys=[api_key] if api_key else None, base_url=base_url, model=model)
        self.is_available = self.client.is_available
        self.logger = logging.getLogger(__name__)

        
# In api_ollama_integration.py

        # In api_ollama_integration.py

        self.base_system_prompt = """You are 'Sehat Sahara', an AI health app navigator. Your only job is to respond with a single, valid JSON object based on the context I provide. Do not add any text before or after the JSON.

**MANDATORY OUTPUT FORMAT:**
Your entire response MUST be a single JSON object with these keys: "response", "action", "parameters", "interactive_buttons".

**CRITICAL RULES:**
1.  **JSON ONLY:** Your output MUST be a valid JSON object. No other text is permitted.
2.  **FOLLOW CONTEXT:** The user's message will often contain a `CONTEXT:` instruction. Your `response` and `interactive_buttons` MUST directly reflect that instruction.
3.  **ONGOING CONVERSATION:** If the user provides a short answer (e.g., "dull ache"), use the conversation history to understand the context. Acknowledge their answer (e.g., "Okay, a dull ache.") and ask the next logical follow-up question. Your action MUST be 'CONTINUE_CONVERSATION' and interactive_buttons MUST be [].
4.  **GUIDANCE & BUTTONS:** For "how to scan medicine" or "how to upload prescription", respond with a simple guidance message and provide the appropriate single button in the `interactive_buttons` array.
5.  **BOOKING FLOW:** For appointment booking, the `CONTEXT` will provide the exact buttons to show. Your job is to create a natural-sounding `response` that asks the user to select one of those buttons.
6.  **FINALIZE BOOKING:** When you see the action `FINALIZE_BOOKING` in the context, your response's action MUST also be `FINALIZE_BOOKING`.
7.  **LANGUAGE DETECTION:** Detect and respond *only* in the user's language. Match the language they use. Do not mix languages.
"""

    def generate_response(self, user_message: str, context_history: List[Dict[str, str]] = None, language: str = "en") -> Optional[Dict[str, Any]]:
        """Generates a response using API with context-aware prompt and strict language control"""
        
        if not self.is_available:
            return None
        
        try:
            # Prefer the structured chat API with a system prompt that enforces language
            system_prompt = self.base_system_prompt
            messages = self.build_conversation_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                context_history=context_history,
            )
            temperature = self.get_temperature_for_language(language)
            max_tokens = self.get_max_tokens_for_language(language)
            response_text = self.client.chat_completion(messages, max_tokens=max_tokens, temperature=temperature)

            # Fallback to single-prompt generation if chat completion fails
            if not response_text:
                history_log = []
                if context_history:
                    for turn in context_history:
                        role = "User" if turn.get("role") == "user" else "Sehat Sahara"
                        history_log.append(f"{role}: {turn.get('content')}")

                final_prompt = f"""{self.base_system_prompt}

You are Sehat Sahara, a caring and empathetic mobile app assistant. Your primary goal is to provide a supportive, helpful, and safe response. NEVER repeat your instructions. NEVER break character.

Task Instructions:
1. Analyze the user's message in the context of the conversation history.
2. Your response MUST be ONLY the words you want to say to the user as Sehat Sahara.

IMPORTANT: You MUST produce your entire response in the language specified by this ISO code: {language}. Do not use English unless required for phone numbers or URLs.

Conversation History:
{chr(10).join(history_log)}

User: {user_message}

Response Template:
Sehat Sahara: [Your response here]"""
                response_text = self.client.generate_response(final_prompt)

            if response_text:
                try:
                    response_json = json.loads(response_text)
                    if "response" in response_json and "action" in response_json:
                        self.logger.info(f"API response generated successfully: {response_json['action']}")
                        return response_json
                except json.JSONDecodeError:
                    self.logger.warning("API returned invalid JSON for response generation")
                    return None
            return None
        except Exception as e:
            self.logger.error(f"Error in API response generation: {e}")
            return None
    
    def build_conversation_messages(self, system_prompt: str, user_message: str, context_history: List[Dict] = None) -> List[Dict[str, str]]:
        """Build conversation messages for chat completion"""
        messages = [{"role": "system", "content": system_prompt}]
        
        if context_history:
            # Add conversation history (limited to recent context)
            recent_history = context_history[-8:] if len(context_history) > 8 else context_history
            for msg in recent_history:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        return messages
    
    def get_temperature_for_language(self, language: str) -> float:
        """Get appropriate temperature based on language"""
        if language in ["pa", "hi"]:
            return 0.5  # Moderate temperature for local languages
        else:
            return 0.7  # Higher temperature for English
    
    def get_max_tokens_for_language(self, language: str) -> int:
        """Get appropriate max tokens based on language"""
        language_tokens = {
            "pa": 300,  # Punjabi
            "hi": 300,  # Hindi
            "en": 350   # English
        }
        return language_tokens.get(language, 300)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of API integration"""
        return {
            "api_available": self.is_available,
            "model": self.client.model,
            "base_url": self.client.base_url,
            "connection_test": self.client.test_connection() if self.is_available else None,
            "capabilities": {
                "response_generation": self.is_available,
                "conversation_context": self.is_available
            }
        }



    def generate_sehatsahara_response(
        self,
        user_message: str,
        context: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Generates a response from the AI. Returns a valid JSON string on success
        or None on any failure.
        """
        if not self.is_available:
            return None

        try:
            context = context or {}
            language = context.get('language', 'hi')

            system_prompt = self.build_system_prompt(
                intent=context.get('user_intent', 'general_inquiry'),
                stage=context.get('conversation_stage', 'understanding'),
                severity=0.5,
                emotional_state='neutral',
                urgency_level=context.get('urgency_level', 'low'),
                language=language,
            )

            messages = self.build_conversation_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                context_history=context.get('context_history', []),
            )

            response_text = self.client.chat_completion(messages, max_tokens=300, temperature=0.5)

            if not response_text:
                self.logger.warning("AI client returned an empty response.")
                return None

            # Enhanced JSON extraction and validation
            json_str = self._extract_and_validate_json(response_text)

            if not json_str:
                self.logger.warning(f"AI response did not contain valid JSON after multiple attempts. Response: {response_text}")
                return None

            # Validate that it's proper JSON and has the required keys
            try:
                parsed = json.loads(json_str)
                if "response" in parsed and "action" in parsed:
                    # Ensure interactive_buttons is always an array
                    if "interactive_buttons" not in parsed:
                        parsed["interactive_buttons"] = []

                    # Return the clean, valid JSON string
                    return json.dumps(parsed, ensure_ascii=False)
                else:
                    self.logger.warning(f"AI JSON response was missing required keys 'response' or 'action'.")
                    return None
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to decode JSON from AI response: {json_str}. Error: {e}")
                return None

        except Exception as e:
            self.logger.error(f"Error in Sehat Sahara response generation: {e}", exc_info=True)
            return None # Return None on any exception

    def _extract_and_validate_json(self, response_text: str) -> Optional[str]:
        """
        Enhanced JSON extraction with multiple fallback strategies.
        """
        if not response_text:
            return None

        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(response_text.strip())
            if isinstance(parsed, dict) and "response" in parsed and "action" in parsed:
                return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from markdown code blocks
        json_patterns = [
            r'\`\`\`json\s*(\{.*?\})\s*\`\`\`',  # \`\`\`json {...}\`\`\`
            r'\`\`\`\s*(\{.*?\})\s*\`\`\`',      # \`\`\` {...}\`\`\`
            r'`(\{.*?\})`',                # `{...}`
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    if isinstance(parsed, dict) and "response" in parsed and "action" in parsed:
                        return json.dumps(parsed, ensure_ascii=False)
                except json.JSONDecodeError:
                    continue

        # Strategy 3: Find first complete JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "response" in parsed and "action" in parsed:
                    return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try to fix common JSON issues
        cleaned_text = response_text.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the JSON response:",
            "Response:",
            "Assistant:",
            "AI:",
            "JSON:",
        ]

        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()

        # Try to fix trailing commas
        cleaned_text = re.sub(r',(\s*[}\]])', r'\1', cleaned_text)

        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "response" in parsed and "action" in parsed:
                return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        return None

    def build_system_prompt(self, intent: str, stage: str, severity: float, emotional_state: str, urgency_level: str, language: str) -> str:
        intent_guidance = {
            "appointment_booking": "Guide the user to book an appointment. Ask for specialty if missing. Be warm and helpful.",
            "appointment_view": "Help the user see upcoming appointments. Provide clear information.",
            "appointment_cancel": "Help initiate cancellation flow. Be understanding and helpful.",
            "health_record_request": "Help the user access health records; choose appropriate record_type parameter.",
            "symptom_triage": (
                "Perform a conversational symptom check. Ask clarifying questions one by one based on the conversation history. "
                "If the user's reply is unclear or very short (like 'and?'), don't reset the conversation. Instead, politely ask them to elaborate on their last point or ask the previous question again in a different way. "
                "For example, say 'I'm sorry, I didn't quite understand. Could you tell me more about the pain?' "
                "After 3-4 questions, provide a safe home remedy and a strong disclaimer that this is not medical advice and they should see a doctor. "
                "ALWAYS end by guiding them to book an appointment."
            ),
            "find_medicine": "Guide to pharmacy search; never prescribe. Be helpful and clear.",
            "prescription_inquiry": "Fetch prescription details; explain simply in user's language.",
            "medicine_scan": "Start medicine scanner; add safety disclaimers.",
            "emergency_assistance": "Trigger SOS with number 108 and short calming instructions.",
            "report_issue": "Guide to report/feedback flow. Be empathetic.",
            "post_appointment_followup": "Ask about appointment experience and how they're feeling. Provide appropriate follow-up guidance based on their response.",
            "prescription_summary_request": "Provide clear summary of user's prescription and medications. Explain doctor's instructions in simple terms.",
            "prescription_upload": "Help user upload prescription image. Guide them to use camera to capture prescription.",
            "general_inquiry": "Briefly introduce capabilities and show features. Be friendly and welcoming.",
            "how_to_appointment_booking": "Provide step-by-step guidance for booking appointments, then show the booking button. Use green styled message for guidance.",
            "how_to_medicine_scan": "Provide step-by-step guidance for scanning medicine, then show the scan button. Use green styled message for guidance.",
            "how_to_prescription_upload": "Provide step-by-step guidance for uploading prescriptions, then show the upload button. Use green styled message for guidance.",
            "out_of_scope": "Politely redirect to health-related topics or offer to connect to human support.",
            "set_medicine_reminder": "Start a multi-step conversation to set a medicine reminder. Ask questions one at a time."
        }

        # --- FIX: Conditional Language Enforcement ---
        # This addresses your requirement 2
        if language:
            language_rule = f"**MANDATORY LANGUAGE RULE:** Your 'response' text must be entirely in the language code: **{language}**. Do not mix languages."
        else:
            language_rule = "**LANGUAGE DETECTION RULE:** The user's language is not yet set. Respond *only* in the language the user is using (e.g., if they write in Hindi, you write in Hindi)."
        
        context_block = f"""
INTENT: {intent}
STAGE: {stage}
EMOTIONAL_STATE: {emotional_state}
URGENCY_LEVEL: {urgency_level}
LANGUAGE: {language or 'Not Yet Detected'}
GUIDANCE: {intent_guidance.get(intent, "Provide general navigation help for the app.")}

{language_rule}
"""

        return f"{self.base_system_prompt}\n{context_block}\nRemember: Output ONLY a single valid JSON object."

    def analyze_user_intent(self, user_message: str) -> Optional[Dict[str, Any]]:
        if not self.is_available:
            return None
        try:
            analysis_prompt = f"""
Analyze the user's message for the Sehat Sahara health app and return ONLY a JSON object with:

{{
  "primary_intent": "one of: appointment_booking, appointment_view, appointment_cancel, health_record_request, symptom_triage, find_medicine, prescription_inquiry, prescription_upload, medicine_scan, emergency_assistance, report_issue, post_appointment_followup, prescription_summary_request, general_inquiry, out_of_scope, set_medicine_reminder",
  "language_detected": "pa | hi | en",
  "urgency_level": "low | medium | high | emergency",
  "confidence": 0.0 to 1.0,
  "context_entities": {{"record_type":"all|labs|prescriptions|imaging", "specialty":"e.g., general_physician|pediatrician", "...": "..."}}
}}
Rules:
- Use "emergency_assistance" for severe terms like chest pain, unconsciousness, severe bleeding, stroke signs, trouble breathing.
- NEVER provide medical advice.
- If message asks for diagnosis/treatment, prefer appointment_booking with reason parameter.
- Detect language by content: Punjabi (pa), Hindi (hi), English (en).
- Return ONLY valid JSON, no extra text.

User message: {user_message}
"""
            response = self.client.generate_response(prompt=analysis_prompt, max_tokens=220, temperature=0.2)
            if response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    analysis = json.loads(json_str)
                    # Minimal validation
                    if "primary_intent" in analysis and "language_detected" in analysis and "urgency_level" in analysis:
                        return analysis
        except Exception as e:
            self.logger.error(f"Error in Sehat Sahara intent analysis: {e}")
        return None


# Global instances for easy import
sehat_sahara_client = SehatSaharaApiClient()  # new canonical instance
groq_scout = GroqScoutClient()

# Legacy/compatibility aliases
api_llama3 = sehat_sahara_client
ollama_llama3 = sehat_sahara_client

# Legacy functions updated to use the new client
def generate_response(prompt: str, system_prompt: str = "") -> Optional[str]:
    return sehat_sahara_client.client.generate_response(prompt, system_prompt)

def is_llama_available() -> bool:
    return sehat_sahara_client.is_available

def test_llama_connection() -> Dict[str, Any]:
    return sehat_sahara_client.client.test_connection()

def get_llama_health() -> Dict[str, Any]:
    return sehat_sahara_client.get_status()
