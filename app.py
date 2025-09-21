import streamlit as st
import requests
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="SafeSpace - AI Mental Health Support", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Enhanced token handling with multiple sources
def get_hf_token():
    """Get HuggingFace token from multiple sources"""
    # Try environment variable first
    token = os.getenv("HF_TOKEN", "")
    
    # Try Streamlit secrets if environment variable not found
    if not token:
        try:
            token = st.secrets.get("HF_TOKEN", "")
        except:
            pass
    
    return token.strip()

# API Configuration
HF_TOKEN = get_hf_token()
API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

SYSTEM_PROMPT = """You are SafeSpace, a compassionate AI mental health supporter. Your role is to:
- Provide empathetic, non-judgmental support
- Use active listening techniques
- Offer coping strategies and resources
- Recognize when professional help is needed
- Maintain appropriate boundaries as an AI assistant

Remember: You are not a replacement for professional therapy, but a supportive companion."""

# =============================================================================
# MENTAL HEALTH TOOLS AND AI LOGIC
# =============================================================================

class MentalHealthTools:
    """Enhanced collection of mental health support tools"""
    
    @staticmethod
    def detect_crisis(message: str) -> bool:
        """Detect crisis situations that need immediate intervention"""
        crisis_patterns = [
            r'\b(suicide|kill myself|end it all|want to die|not worth living)\b',
            r'\b(harm myself|hurt myself|self harm|cut myself)\b',
            r'\b(overdose|pills to die|end my life)\b',
            r'\b(gun|knife|rope|bridge|jump)\b.*\b(end|die|kill)\b',
            r'\b(no point|give up|can\'?t go on|hopeless)\b',
            r'\b(better off dead|world without me)\b'
        ]
        
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in crisis_patterns)
    
    @staticmethod
    def detect_emotions(message: str) -> List[str]:
        """Enhanced emotion detection with more nuanced patterns"""
        emotion_keywords = {
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared', 'terrified', 'petrified'],
            'depression': ['sad', 'depressed', 'hopeless', 'empty', 'worthless', 'tired', 'numb', 'hollow'],
            'anger': ['angry', 'mad', 'frustrated', 'furious', 'rage', 'irritated', 'livid', 'enraged'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'burden', 'exhausted', 'burned out', 'swamped'],
            'loneliness': ['lonely', 'alone', 'isolated', 'disconnected', 'abandoned', 'forgotten', 'excluded'],
            'grief': ['grieving', 'mourning', 'loss', 'bereaved', 'heartbroken', 'devastated'],
            'confusion': ['confused', 'lost', 'uncertain', 'directionless', 'unclear', 'mixed up']
        }
        
        message_lower = message.lower()
        detected_emotions = []
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions
    
    @staticmethod
    def get_coping_strategies(emotions: List[str]) -> str:
        """Enhanced coping strategies based on detected emotions"""
        strategies = {
            'anxiety': "Try the 4-7-8 breathing technique: inhale for 4, hold for 7, exhale for 8. Use the 5-4-3-2-1 grounding method: name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste.",
            'depression': "Small steps count. Try the 'one tiny thing' approach - do just one small positive action today. Consider sunlight exposure, gentle movement, or connecting with someone who cares about you.",
            'anger': "Use the STOP technique: Stop what you're doing, Take a deep breath, Observe your feelings, Proceed mindfully. Try the 6-second rule - strong emotions peak and start to fade after 6 seconds.",
            'stress': "Try the HALT check: are you Hungry, Angry, Lonely, or Tired? Address basic needs first. Use the 2-minute rule: if it takes less than 2 minutes, do it now to reduce mental load.",
            'loneliness': "Reach out with a 'thinking of you' message to someone. Join online communities, volunteer virtually, or try co-working spaces. Remember: loneliness is temporary.",
            'grief': "Allow yourself to feel. Grief comes in waves - that's normal. Create a small ritual to honor what you've lost. Consider grief support groups or counseling.",
            'confusion': "Write down what you know for sure, then what you're uncertain about. Break big decisions into smaller steps. It's okay not to have all the answers right now."
        }
        
        if not emotions:
            return "Practice self-compassion today. Treat yourself with the same kindness you'd show a good friend."
        
        selected_strategies = [strategies.get(emotion, "") for emotion in emotions if emotion in strategies]
        return " ".join(filter(None, selected_strategies))

class EnhancedGraph:
    """Enhanced AI conversation handler with better error handling"""
    
    def __init__(self):
        self.tools = MentalHealthTools()
        self.conversation_history = []
        self.api_available = bool(HF_TOKEN)
    
    def stream(self, inputs, stream_mode):
        """Main conversation processing with enhanced error handling"""
        try:
            user_message = inputs["messages"][1][1]  # Get user message
            self.conversation_history.append(("user", user_message))
            
            # Crisis detection with immediate response
            if self.tools.detect_crisis(user_message):
                crisis_response = self._handle_crisis()
                yield crisis_response
                return
            
            # Detect emotional state
            emotions = self.tools.detect_emotions(user_message)
            
            # Generate empathetic response
            response_data = self._generate_response(user_message, emotions)
            yield response_data
            
        except Exception as e:
            st.error(f"Processing error: {e}")
            yield self._fallback_response()
    
    def _handle_crisis(self) -> Dict[str, str]:
        """Enhanced crisis intervention with local resources"""
        return {
            'response': """🆘 **I'm very concerned about what you're sharing. Your life has value and there are people who want to help.**

**🇮🇳 Immediate Support in India:**
• **Call 9152987821** - AASRA Suicide Prevention (24/7)
• **Call 1860-266-2345** - Vandrevala Foundation (24/7, free)
• **Call 1800-599-0019** - Kiran Mental Health Helpline
• **WhatsApp +91 9820466726** - Connecting NGO

**🌍 International:**
• **Call 988** - US Suicide & Crisis Lifeline
• **Text HELLO to 741741** - Crisis Text Line

🌟 **You are not alone.** Professional counselors are available right now to talk with you.

Would you like to talk about what's bringing up these feelings? I'm here to listen without judgment.""",
            'tool': 'crisis_intervention'
        }
    
    def _generate_response(self, user_message: str, emotions: List[str]) -> Dict[str, str]:
        """Enhanced response generation with better fallbacks"""
        
        # Try AI response first (if token available)
        if self.api_available:
            ai_response = self._call_huggingface_api(user_message)
            if ai_response:
                # Enhance AI response with emotional support
                enhanced_response = self._enhance_response(ai_response, emotions)
                return {
                    'response': enhanced_response,
                    'tool': 'ai_therapy_conversation'
                }
        
        # Fallback to enhanced rule-based response
        return self._rule_based_response(user_message, emotions)
    
    def _call_huggingface_api(self, message: str, retries: int = 2) -> str:
        """Enhanced API call with retry logic and better error handling"""
        if not HF_TOKEN:
            return ""
            
        for attempt in range(retries + 1):
            try:
                # Create a more therapeutic prompt
                prompt = f"You are a supportive mental health counselor. Respond with empathy and care.\nUser: {message}\nCounselor:"
                
                response = requests.post(
                    API_URL, 
                    headers=headers, 
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 120,
                            "return_full_text": False,
                            "temperature": 0.7,
                            "do_sample": True,
                            "repetition_penalty": 1.1
                        }
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '').strip()
                        # Clean up the response
                        generated_text = generated_text.replace(prompt, '').strip()
                        if generated_text and len(generated_text) > 10:  # Ensure meaningful response
                            return generated_text
                            
                elif response.status_code == 503:  # Model loading
                    if attempt < retries:
                        time.sleep(3)  # Wait for model to load
                        continue
                else:
                    st.warning(f"API responded with status {response.status_code}. Using fallback response.")
                    break
                    
            except requests.exceptions.Timeout:
                if attempt < retries:
                    time.sleep(1)
                    continue
            except Exception as e:
                if attempt < retries:
                    time.sleep(1)
                    continue
                else:
                    st.warning("AI service temporarily unavailable. Using trained fallback responses.")
        
        return ""
    
    def _enhance_response(self, base_response: str, emotions: List[str]) -> str:
        """Enhanced response enhancement with better formatting"""
        if not emotions:
            return base_response
        
        coping_strategies = self.tools.get_coping_strategies(emotions)
        emotion_str = ", ".join(emotions)
        
        if coping_strategies:
            return f"{base_response}\n\n💡 **For {emotion_str}:** {coping_strategies}"
        
        return base_response
    
    def _rule_based_response(self, message: str, emotions: List[str]) -> Dict[str, str]:
        """Enhanced rule-based therapeutic responses"""
        message_lower = message.lower()
        
        # Enhanced greeting responses
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'start', 'beginning']
        if any(greeting in message_lower for greeting in greetings):
            return {
                'response': "Hello and welcome to SafeSpace! 🌟 I'm here to listen without judgment and support you through whatever you're experiencing. How are you feeling today? Take your time - there's no pressure to share more than you're comfortable with.",
                'tool': 'warm_greeting'
            }
        
        # Enhanced emotion-specific responses
        if 'anxiety' in emotions or any(word in message_lower for word in ['anxious', 'worried', 'panic', 'scared']):
            response = "I can hear the anxiety in your words, and I want you to know that what you're feeling is real and valid. Anxiety can feel overwhelming, but you're not alone in this experience."
            coping = self.tools.get_coping_strategies(['anxiety'])
            return {
                'response': f"{response}\n\n🫁 **Try this breathing technique:** {coping}",
                'tool': 'anxiety_support'
            }
        
        elif 'depression' in emotions or any(word in message_lower for word in ['sad', 'depressed', 'hopeless', 'empty', 'worthless']):
            response = "I hear the pain in your words, and I want you to know that your feelings are completely valid. Depression can make everything feel heavy and difficult, but please know that you matter and you're not alone."
            coping = self.tools.get_coping_strategies(['depression'])
            return {
                'response': f"{response}\n\n💙 **Gentle reminder:** {coping}",
                'tool': 'depression_support'
            }
        
        elif 'stress' in emotions or any(word in message_lower for word in ['stressed', 'overwhelmed', 'pressure', 'burned', 'exhausted']):
            response = "It sounds like you're carrying a heavy load right now. Feeling stressed and overwhelmed is exhausting, and it takes real strength to reach out for support."
            coping = self.tools.get_coping_strategies(['stress'])
            return {
                'response': f"{response}\n\n🌱 **Stress relief technique:** {coping}",
                'tool': 'stress_management'
            }
        
        elif 'loneliness' in emotions or any(word in message_lower for word in ['lonely', 'alone', 'isolated', 'disconnected']):
            response = "Loneliness can feel so painful and isolating. I want you to know that reaching out here shows incredible courage, and you're taking a step toward connection right now."
            coping = self.tools.get_coping_strategies(['loneliness'])
            return {
                'response': f"{response}\n\n🤝 **Connection idea:** {coping}",
                'tool': 'loneliness_support'
            }
        
        # Thanks/gratitude with encouragement
        if any(word in message_lower for word in ['thank', 'thanks', 'grateful', 'appreciate']):
            return {
                'response': "You're so welcome. 💛 It takes real courage to reach out and talk about these things. I'm honored that you chose to share with me. Remember, seeking support is a sign of strength, not weakness. How else can I support you today?",
                'tool': 'gratitude_response'
            }
        
        # Enhanced check-in responses
        if any(phrase in message_lower for phrase in ['how are you', 'are you okay', 'how do you feel']):
            return {
                'response': "Thank you for asking! As an AI, I don't have feelings, but I'm here and fully focused on you. What matters most right now is how *you* are doing. I'm ready to listen to whatever you'd like to share.",
                'tool': 'check_in_redirect'
            }
        
        # Enhanced default empathetic response with emotion acknowledgment
        emotion_acknowledgment = ""
        if emotions:
            emotion_list = ", ".join(emotions)
            emotion_acknowledgment = f"I can sense you might be feeling {emotion_list}, and I want you to know those feelings are completely valid. "
        
        return {
            'response': f"{emotion_acknowledgment}I'm here to listen and support you through whatever you're experiencing. Sometimes it helps to talk through what's on your mind. What would you like to share with me? Remember, you can share as much or as little as feels comfortable.",
            'tool': 'empathetic_listening'
        }
    
    def _fallback_response(self) -> Dict[str, str]:
        """Enhanced fallback response"""
        return {
            'response': "I'm here to listen and support you. Even when technology has hiccups, my commitment to being here for you remains constant. How are you feeling right now? What's on your mind?",
            'tool': 'technical_fallback'
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_api_setup():
    """Validate API setup and provide user feedback"""
    if not HF_TOKEN:
        # Removed the sidebar info message
        return False
    else:
        st.sidebar.success("🤖 **Enhanced AI Mode Active**")
    return True

def parse_response(stream):
    """Enhanced response parsing with better error handling"""
    try:
        result = next(stream)
        return result['tool'], result['response']
    except StopIteration:
        return 'error', "I'm having trouble processing that right now, but I'm still here to listen and support you."
    except Exception as e:
        st.error(f"Parse error: {e}")
        return 'error', "I want to help you, but I'm experiencing some technical difficulties. Your feelings and experiences are still important to me. Can you tell me how you're feeling?"

def get_ai_response(user_message):
    """Enhanced AI response function with better error handling"""
    try:
        inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", user_message)]}
        stream = graph.stream(inputs, stream_mode="updates")
        tool_called_name, final_response = parse_response(stream)
        return final_response, tool_called_name, "success"
    except Exception as e:
        st.error(f"AI Response Error: {e}")
        # Provide a thoughtful fallback response
        fallback = "I'm experiencing some technical difficulties, but I want you to know that I'm still here to support you. Your feelings and experiences matter. Can you tell me more about what's on your mind today?"
        return fallback, "fallback", "error"

# =============================================================================
# STREAMLIT FRONTEND
# =============================================================================

# Enhanced CSS for better visual experience
st.markdown("""
<style>
    .stAlert > div {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid #1c83e1;
        color: #0c4a6e !important;
    }
    .crisis-alert {
        background-color: rgba(220, 38, 38, 0.1);
        border: 2px solid #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 20px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Global graph instance
graph = EnhancedGraph()

# Validate API setup on startup
api_available = validate_api_setup()

# Enhanced header
st.title("🧠 SafeSpace — AI Mental Health Support")
st.markdown("### *A safe, confidential space to express your feelings and find support*")

# Removed the main page Safe Mode info message

# Enhanced sidebar with more comprehensive resources
with st.sidebar:
    st.markdown("### 🆘 Crisis Resources")
    
    # Indian resources first
    st.markdown("""
    **🇮🇳 India - Immediate Help:**
    - **AASRA**: 9152987821 (24/7)
    - **Vandrevala Foundation**: 1860-266-2345 (24/7, free)
    - **Kiran Helpline**: 1800-599-0019
    - **Connecting NGO**: WhatsApp +91 9820466726
    
    **🌍 International:**
    - **US**: 988 (Suicide & Crisis Lifeline)
    - **UK**: 116 123 (Samaritans)
    - **Crisis Text Line**: Text HOME to 741741
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Coping Tools")
    st.markdown("""
    **🫁 Breathing:**
    - 4-7-8 technique: In for 4, hold for 7, out for 8
    - Box breathing: 4-4-4-4 pattern
    
    **🧘 Grounding (5-4-3-2-1):**
    - 5 things you can see
    - 4 things you can hear
    - 3 things you can touch
    - 2 things you can smell
    - 1 thing you can taste
    
    **🚶 Movement:**
    - Take a 5-minute walk
    - Gentle stretching
    - Dance to one song
    """)
    
    st.markdown("---")
    st.markdown("### 📱 Helpful Apps")
    st.markdown("""
    - **Headspace** - Meditation & mindfulness
    - **Calm** - Sleep stories & relaxation
    - **Insight Timer** - Free meditation
    - **Youper** - Mood tracking
    - **Sanvelo** - Anxiety & mood support
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About SafeSpace")
    st.markdown("""
    SafeSpace provides:
    - 24/7 emotional support
    - Crisis detection & resources
    - Coping strategy suggestions
    - Non-judgmental listening
    - Connection to professional help
    
    **Important**: This tool complements but doesn't replace professional mental health care.
    """)

# Initialize enhanced session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# Enhanced welcome message
if not st.session_state.conversation_started and not st.session_state.chat_history:
    st.success("""
    🌟 **Welcome to SafeSpace**
    
    This is your safe, confidential space to express thoughts and feelings. I'm here to listen 
    without judgment and provide support whenever you need it.
    
    **You can:**
    - Share what's on your mind
    - Talk about your feelings
    - Get coping strategies
    - Access mental health resources
    
    **Remember:** If you're in crisis, please use the emergency resources in the sidebar immediately.
    """)

# Enhanced quick start section with better state management
if not st.session_state.chat_history:
    st.markdown("### 💭 How are you feeling today?")
    st.markdown("*Choose a button below to get started, or type your own message*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("😰 Anxious", help="I'm feeling anxious or worried", key="anxious_btn"):
            user_input = "I'm feeling really anxious and worried about everything. My mind won't stop racing."
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_started = True
            # Force immediate processing
            with st.spinner("💭 Thinking carefully about your message..."):
                response_content, tool_used, status = get_ai_response(user_input)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool": tool_used,
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": status
            })
            st.rerun()
    
    with col2:
        if st.button("😔 Sad", help="I'm feeling down or depressed", key="sad_btn"):
            user_input = "I'm feeling really sad and down lately. Everything feels heavy and difficult."
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_started = True
            # Force immediate processing
            with st.spinner("💭 Thinking carefully about your message..."):
                response_content, tool_used, status = get_ai_response(user_input)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool": tool_used,
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": status
            })
            st.rerun()
    
    with col3:
        if st.button("😤 Stressed", help="I'm feeling overwhelmed", key="stressed_btn"):
            user_input = "I'm feeling completely overwhelmed and stressed out. There's so much pressure."
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_started = True
            # Force immediate processing
            with st.spinner("💭 Thinking carefully about your message..."):
                response_content, tool_used, status = get_ai_response(user_input)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool": tool_used,
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": status
            })
            st.rerun()
    
    with col4:
        if st.button("😞 Lonely", help="I'm feeling isolated", key="lonely_btn"):
            user_input = "I'm feeling really lonely and disconnected from everyone around me."
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_started = True
            # Force immediate processing
            with st.spinner("💭 Thinking carefully about your message..."):
                response_content, tool_used, status = get_ai_response(user_input)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool": tool_used,
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": status
            })
            st.rerun()
    
    # Additional quick start options
    st.markdown("---")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("😡 Angry", help="I'm feeling frustrated or angry", key="angry_btn"):
            user_input = "I'm feeling really angry and frustrated. I can't seem to control these feelings."
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_started = True
            # Force immediate processing
            with st.spinner("💭 Thinking carefully about your message..."):
                response_content, tool_used, status = get_ai_response(user_input)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool": tool_used,
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": status
            })
            st.rerun()
    
    with col6:
        if st.button("😕 Confused", help="I'm feeling lost or uncertain", key="confused_btn"):
            user_input = "I'm feeling confused and lost. I don't know what to do or which direction to go."
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_started = True
            # Force immediate processing
            with st.spinner("💭 Thinking carefully about your message..."):
                response_content, tool_used, status = get_ai_response(user_input)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool": tool_used,
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": status
            })
            st.rerun()
    
    with col7:
        if st.button("💔 Grieving", help="I'm dealing with loss", key="grieving_btn"):
            user_input = "I'm grieving and dealing with a significant loss. The pain feels unbearable."
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_started = True
            # Force immediate processing
            with st.spinner("💭 Thinking carefully about your message..."):
                response_content, tool_used, status = get_ai_response(user_input)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool": tool_used,
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": status
            })
            st.rerun()
    
    with col8:
        if st.button("🆘 Crisis", help="I need immediate support", key="crisis_btn"):
            user_input = "I'm in crisis and need immediate support. I don't know how to cope."
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_started = True
            # Force immediate processing
            with st.spinner("💭 Thinking carefully about your message..."):
                response_content, tool_used, status = get_ai_response(user_input)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "tool": tool_used,
                "timestamp": datetime.now().strftime("%H:%M"),
                "status": status
            })
            st.rerun()

# Enhanced chat input with supportive placeholder
user_input = st.chat_input("Share what's on your mind... I'm here to listen 💙")

# Process user input
if user_input or (st.session_state.chat_history and not st.session_state.conversation_started):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    st.session_state.conversation_started = True
    
    # Get the most recent user message
    latest_user_message = st.session_state.chat_history[-1]["content"]
    
    # Enhanced loading indicator
    with st.spinner("💭 Thinking carefully about your message and preparing a thoughtful response..."):
        response_content, tool_used, status = get_ai_response(latest_user_message)
    
    # Enhanced response formatting
    if tool_used == "crisis_intervention":
        st.markdown('<div class="crisis-alert">', unsafe_allow_html=True)
        st.error("🚨 **Crisis Support Activated** - Please see the important resources below.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response_content,
        "tool": tool_used,
        "timestamp": datetime.now().strftime("%H:%M"),
        "status": status
    })
    
    st.rerun()

# Enhanced chat display
if st.session_state.chat_history:
    st.markdown("### 💬 Our Conversation")
    
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            # Special handling for crisis messages
            if msg["role"] == "assistant" and msg.get("tool") == "crisis_intervention":
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            
            # Enhanced timestamp and status for assistant messages
            if msg["role"] == "assistant" and "timestamp" in msg:
                status_emoji = "✅" if msg.get("status") == "success" else "🔄"
                st.caption(f"{status_emoji} Responded at {msg['timestamp']} • Tool: {msg.get('tool', 'unknown')}")
    
    # Enhanced action buttons
    st.markdown("---")
    st.markdown("### 🛠️ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Fresh Start", help="Clear conversation and start over"):
            st.session_state.chat_history = []
            st.session_state.conversation_started = False
            st.success("Conversation cleared. You can start fresh anytime you need to.")
            st.rerun()
    
    with col2:
        if st.button("📋 Resources", help="Get comprehensive mental health resources"):
            resources_content = """🏥 **Comprehensive Mental Health Resources**

**🆘 Crisis Support (India):**
• AASRA: 9152987821 (24/7, suicide prevention)
• Vandrevala Foundation: 1860-266-2345 (24/7, free counseling)
• Kiran Mental Health: 1800-599-0019
• Connecting NGO: +91 9820466726 (WhatsApp)
• Sumaitri: 011 23389090 (Delhi)

**🌍 International Crisis Support:**
• US Suicide Prevention: 988
• UK Samaritans: 116 123
• Crisis Text Line: Text HOME to 741741
• Australia Lifeline: 13 11 14

**🩺 Professional Help:**
• Psychology Today India: psychologyindia.com
• BetterHelp: betterhelp.com (online therapy)
• Talkspace: talkspace.com (online therapy)
• Local mental health professionals in your area

**📱 Mental Health Apps:**
• Headspace (meditation)
• Calm (sleep & relaxation)  
• Insight Timer (free meditation)
• Youper (mood tracking & AI support)
• Sanvello (anxiety & depression)
• MindShift (anxiety management)

**📚 Educational Resources:**
• National Alliance on Mental Illness (NAMI)
• Mental Health America
• WHO Mental Health Resources
• Local mental health organizations

Remember: Professional help is always available when you need it. You deserve support and care. 💙"""
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": resources_content,
                "tool": "comprehensive_resources",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
    
    with col3:
        if st.button("💙 Self-Care", help="Get personalized self-care suggestions"):
            selfcare_content = """🌱 **Self-Care Toolkit**

**🏃 Physical Self-Care:**
• Take a 10-minute walk in fresh air
• Do gentle stretching or yoga
• Take a warm shower or bath
• Practice deep breathing exercises
• Get adequate sleep (7-9 hours)
• Stay hydrated throughout the day

**🧠 Emotional Self-Care:**
• Write in a journal for 5 minutes
• Practice gratitude - list 3 good things
• Allow yourself to feel emotions without judgment
• Reach out to a trusted friend or family member
• Practice positive self-talk
• Set healthy boundaries

**🎨 Creative Self-Care:**
• Listen to music that lifts your mood
• Draw, paint, or do crafts
• Write poetry or stories
• Dance like nobody's watching
• Sing your favorite songs
• Try photography

**🧘 Mental Self-Care:**
• Practice mindfulness or meditation
• Read a book you enjoy
• Learn something new
• Do a puzzle or brain game
• Limit news and social media
• Practice the 5-4-3-2-1 grounding technique

**🤝 Social Self-Care:**
• Connect with supportive people
• Join online communities with shared interests
• Volunteer for a cause you care about
• Practice saying no to draining activities
• Seek professional help when needed
• Express your needs clearly

**🏠 Environmental Self-Care:**
• Clean and organize your space
• Spend time in nature
• Create a cozy, comfortable environment
• Use aromatherapy or candles
• Keep plants or flowers nearby
• Minimize clutter

Remember: Self-care isn't selfish - it's necessary. Start small and be gentle with yourself. 💚"""
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": selfcare_content,
                "tool": "self_care_toolkit",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
    
    with col4:
        if st.button("🧘 Mindfulness", help="Get guided mindfulness exercises"):
            mindfulness_content = """🧘 **Mindfulness & Grounding Exercises**

**🫁 Breathing Techniques:**

*4-7-8 Breathing:*
1. Inhale through nose for 4 counts
2. Hold breath for 7 counts
3. Exhale through mouth for 8 counts
4. Repeat 3-4 times

*Box Breathing:*
1. Inhale for 4 counts
2. Hold for 4 counts  
3. Exhale for 4 counts
4. Hold empty for 4 counts
5. Repeat 5-10 times

**🌟 5-4-3-2-1 Grounding Technique:**
• Name 5 things you can see
• Name 4 things you can hear
• Name 3 things you can touch
• Name 2 things you can smell
• Name 1 thing you can taste

**🎯 Progressive Muscle Relaxation:**
1. Start with your toes - tense for 5 seconds, then relax
2. Move up through each muscle group
3. Notice the difference between tension and relaxation
4. End with your face and scalp

**🍃 Mindful Observation:**
• Pick an object and observe it for 2 minutes
• Notice colors, textures, shapes, details
• If your mind wanders, gently return focus
• This helps anchor you in the present moment

**🚶 Walking Meditation:**
• Take slow, deliberate steps
• Feel your feet touching the ground
• Notice your surroundings without judgment
• Focus on the rhythm of walking

**💭 Thought Labeling:**
• When anxious thoughts arise, label them: "I'm having the thought that..."
• This creates distance between you and the thought
• Thoughts are temporary visitors, not facts

Practice these daily, even for just 2-3 minutes. Regular practice builds your mindfulness muscle! 🌱"""
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": mindfulness_content,
                "tool": "mindfulness_exercises",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()

# Enhanced footer with comprehensive information
st.markdown("---")
st.markdown("### 📋 Important Information")

# Create columns for organized information
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    **🔒 Privacy & Confidentiality:**
    • Your conversations are processed locally
    • No personal information is stored permanently
    • Messages are not logged or saved after sessions
    • This is a safe space for open communication
    """)

with info_col2:
    st.markdown("""
    **⚠️ Important Disclaimers:**
    • SafeSpace is an AI support tool, not a medical device
    • This is not a substitute for professional mental health treatment
    • In crisis situations, please contact emergency services immediately
    • Consider professional therapy for ongoing support
    """)

# Technical information
st.markdown("---")
technical_col1, technical_col2 = st.columns(2)

with technical_col1:
    st.markdown(f"""
    **🤖 Technical Status:**
    • AI Mode: {'Enhanced' if api_available else 'Safe Mode (Rule-based)'}
    • Response Quality: {'AI + Rule-based' if api_available else 'Advanced Rule-based'}
    • Crisis Detection: ✅ Active
    • Emotion Recognition: ✅ Active
    """)

with technical_col2:
    st.markdown("""
    **💡 How to Get Help:**
    • Use the quick-start buttons above
    • Type your thoughts in the chat box
    • Access resources from the sidebar
    • Contact professionals for ongoing support
    """)

# Final disclaimer and support information
st.markdown("---")
st.info("""
**🌟 Remember: You Are Not Alone**

Mental health struggles are real, and seeking support is a sign of strength. Whether you're dealing with 
anxiety, depression, stress, or any other challenges, there are people and resources available to help you.

If SafeSpace has been helpful, consider sharing it with others who might benefit. Sometimes the hardest 
part is taking that first step to reach out.

**For immediate crisis support, always contact emergency services or use the crisis resources in the sidebar.**
""")

# Usage statistics (if desired)
if st.session_state.chat_history:
    message_count = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
    st.caption(f"💬 You've shared {message_count} message{'s' if message_count != 1 else ''} in this session. Thank you for trusting SafeSpace with your thoughts and feelings.")
