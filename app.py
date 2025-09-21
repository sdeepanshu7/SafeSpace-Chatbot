import streamlit as st
import requests
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

# =============================================================================
# AI AGENT LOGIC (Combined from ai_agent.py)
# =============================================================================

# Get free token at: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Use environment variable for security
API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

SYSTEM_PROMPT = """You are SafeSpace, a compassionate AI mental health supporter. Your role is to:
- Provide empathetic, non-judgmental support
- Use active listening techniques
- Offer coping strategies and resources
- Recognize when professional help is needed
- Maintain appropriate boundaries as an AI assistant

Remember: You are not a replacement for professional therapy, but a supportive companion."""

class MentalHealthTools:
    """Collection of mental health support tools"""
    
    @staticmethod
    def detect_crisis(message: str) -> bool:
        """Detect crisis situations that need immediate intervention"""
        crisis_patterns = [
            r'\b(suicide|kill myself|end it all|want to die|not worth living)\b',
            r'\b(harm myself|hurt myself|self harm|cut myself)\b',
            r'\b(overdose|pills to die|end my life)\b',
            r'\b(gun|knife|rope|bridge|jump)\b.*\b(end|die|kill)\b'
        ]
        
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in crisis_patterns)
    
    @staticmethod
    def detect_emotions(message: str) -> List[str]:
        """Detect emotional states in user message"""
        emotion_keywords = {
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared'],
            'depression': ['sad', 'depressed', 'hopeless', 'empty', 'worthless', 'tired'],
            'anger': ['angry', 'mad', 'frustrated', 'furious', 'rage', 'irritated'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'burden', 'exhausted'],
            'loneliness': ['lonely', 'alone', 'isolated', 'disconnected', 'abandoned']
        }
        
        message_lower = message.lower()
        detected_emotions = []
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions
    
    @staticmethod
    def get_coping_strategies(emotions: List[str]) -> str:
        """Provide coping strategies based on detected emotions"""
        strategies = {
            'anxiety': "Try deep breathing: inhale for 4, hold for 4, exhale for 6. Grounding technique: name 5 things you see, 4 you hear, 3 you touch.",
            'depression': "Small steps matter. Try one tiny positive action today. Consider reaching out to a friend or doing something you used to enjoy.",
            'anger': "Take a pause before reacting. Try physical release like walking or squeezing a stress ball. Count to 10 and breathe deeply.",
            'stress': "Prioritize your tasks. Take breaks every hour. Try progressive muscle relaxation: tense and release each muscle group.",
            'loneliness': "Consider reaching out to one person today. Join online communities or local groups with shared interests."
        }
        
        if not emotions:
            return "Remember to practice self-care. Take time for activities that bring you peace."
        
        return " ".join([strategies.get(emotion, "") for emotion in emotions if emotion in strategies])

class Graph:
    def __init__(self):
        self.tools = MentalHealthTools()
        self.conversation_history = []
    
    def stream(self, inputs, stream_mode):
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
            print(f"Error in stream: {e}")
            yield self._fallback_response()
    
    def _handle_crisis(self) -> Dict[str, str]:
        """Handle crisis situations with immediate resources"""
        return {
            'response': """I'm very concerned about what you're sharing. Your life has value and there are people who want to help.

🆘 **Immediate Support:**
• **Call 988** - Suicide & Crisis Lifeline (24/7, free, confidential)
• **Text HOME to 741741** - Crisis Text Line
• **Call 911** for immediate emergencies

🌟 **You are not alone.** Professional counselors are available right now to talk with you.

Would you like to talk about what's bringing up these feelings?""",
            'tool': 'crisis_intervention'
        }
    
    def _generate_response(self, user_message: str, emotions: List[str]) -> Dict[str, str]:
        """Generate therapeutic response using DialoGPT or fallback"""
        
        # Try DialoGPT first (if token available)
        if HF_TOKEN:
            ai_response = self._call_huggingface_api(user_message)
            if ai_response:
                # Enhance AI response with emotional support
                enhanced_response = self._enhance_response(ai_response, emotions)
                return {
                    'response': enhanced_response,
                    'tool': 'ai_therapy_conversation'
                }
        
        # Fallback to rule-based response
        return self._rule_based_response(user_message, emotions)
    
    def _call_huggingface_api(self, message: str) -> str:
        """Call HuggingFace API for conversational response"""
        try:
            # Create a more therapeutic prompt
            prompt = f"You are a supportive therapist. User: {message}\nTherapist:"
            
            response = requests.post(
                API_URL, 
                headers=headers, 
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 150,
                        "return_full_text": False,
                        "temperature": 0.7,
                        "do_sample": True
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
            
        except Exception as e:
            print(f"HuggingFace API error: {e}")
        
        return ""
    
    def _enhance_response(self, base_response: str, emotions: List[str]) -> str:
        """Enhance AI response with emotional support and coping strategies"""
        if not emotions:
            return base_response
        
        coping_strategies = self.tools.get_coping_strategies(emotions)
        
        if coping_strategies:
            return f"{base_response}\n\n💡 **Helpful strategy:** {coping_strategies}"
        
        return base_response
    
    def _rule_based_response(self, message: str, emotions: List[str]) -> Dict[str, str]:
        """Generate rule-based therapeutic response"""
        message_lower = message.lower()
        
        # Greeting responses
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in message_lower for greeting in greetings):
            return {
                'response': "Hello! Welcome to SafeSpace. I'm here to listen and support you. How are you feeling today?",
                'tool': 'greeting'
            }
        
        # Emotion-specific responses
        if 'anxiety' in emotions or 'anxious' in message_lower:
            response = "I hear that you're feeling anxious. That's really difficult to experience. Can you tell me what might be contributing to these anxious feelings?"
            coping = self.tools.get_coping_strategies(['anxiety'])
            return {
                'response': f"{response}\n\n💡 **Try this:** {coping}",
                'tool': 'anxiety_support'
            }
        
        elif 'depression' in emotions or any(word in message_lower for word in ['sad', 'depressed', 'hopeless']):
            response = "I can sense that you're going through a really tough time right now. Your feelings are valid, and I want you to know that you're not alone."
            coping = self.tools.get_coping_strategies(['depression'])
            return {
                'response': f"{response}\n\n💙 **Gentle reminder:** {coping}",
                'tool': 'depression_support'
            }
        
        elif 'stress' in emotions or 'stressed' in message_lower or 'overwhelmed' in message_lower:
            response = "It sounds like you're carrying a lot right now. Feeling stressed and overwhelmed is exhausting. What's been weighing on your mind the most?"
            coping = self.tools.get_coping_strategies(['stress'])
            return {
                'response': f"{response}\n\n🌱 **Stress relief:** {coping}",
                'tool': 'stress_management'
            }
        
        # Thanks/gratitude
        if any(word in message_lower for word in ['thank', 'thanks', 'grateful']):
            return {
                'response': "You're very welcome. It takes courage to reach out and talk about these things. I'm glad I could be here for you. How else can I support you today?",
                'tool': 'gratitude_response'
            }
        
        # Default empathetic response
        return {
            'response': "I hear you, and I want you to know that your feelings are valid. Sometimes it helps to talk through what's on your mind. What would you like to share with me?",
            'tool': 'active_listening'
        }
    
    def _fallback_response(self) -> Dict[str, str]:
        """Fallback response when all else fails"""
        return {
            'response': "I'm here to listen and support you. Sometimes technical issues happen, but that doesn't change my commitment to being here for you. How are you feeling right now?",
            'tool': 'technical_fallback'
        }

# Global instance
graph = Graph()

def parse_response(stream):
    """Parse the response from the graph stream"""
    try:
        result = next(stream)
        return result['tool'], result['response']
    except StopIteration:
        return 'error', "I'm having trouble processing that right now, but I'm still here to listen."
    except Exception as e:
        print(f"Parse error: {e}")
        return 'error', "I want to help you, but I'm experiencing some technical difficulties. Can you tell me how you're feeling?"

# =============================================================================
# STREAMLIT FRONTEND (Combined from frontend.py)
# =============================================================================

# Page configuration
st.set_page_config(
    page_title="SafeSpace - AI Mental Health Support", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Minimal CSS for better visibility
st.markdown("""
<style>
    .stAlert > div {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid #1c83e1;
        color: #0c4a6e !important;
    }
</style>
""", unsafe_allow_html=True)

# Simple header without complex CSS
st.title("🧠 SafeSpace — AI Mental Health Support")
st.markdown("### *A safe, confidential space to express your feelings and find support*")

# Sidebar with resources and information
with st.sidebar:
    st.markdown("### 🆘 Crisis Resources")
    st.markdown("""
    **If you're in immediate danger:**
    - **Call 112** for emergencies
    - **Call 1800-121-3667** - Suicide & Crisis Lifeline
    - **Text HOME to 741741** - Crisis Text Line
    
    **Other Resources:**
    - The India Mental Health Alliance (IMHA): 800-950-6264
    - Crisis Text Line: Text HOME to 741741
    - SPIF – Suicide Prevention India Foundation: https://spif.in/
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    - **Deep Breathing**: 4 counts in, 4 hold, 6 out
    - **Grounding**: Name 5 things you see, 4 you hear, 3 you touch
    - **Movement**: Take a short walk or stretch
    - **Connection**: Reach out to someone you trust
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About SafeSpace")
    st.markdown("""
    SafeSpace is an AI-powered mental health support tool designed to provide:
    - **24/7 availability** for when you need to talk
    - **Non-judgmental listening** and support
    - **Crisis detection** and resource connection
    - **Coping strategies** and emotional support
    
    **Important**: This is not a replacement for professional therapy or medical treatment.
    """)

# Initialize chat history and session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# AI Response Function (replaces backend call)
def get_ai_response(user_message):
    """Get AI response using the integrated AI agent"""
    try:
        inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", user_message)]}
        stream = graph.stream(inputs, stream_mode="updates")
        tool_called_name, final_response = parse_response(stream)
        return final_response, tool_called_name, "success"
    except Exception as e:
        print(f"AI Error: {e}")
        return "I'm here to listen and support you. How are you feeling right now?", "fallback", "error"

# Welcome message for first-time users
if not st.session_state.conversation_started and not st.session_state.chat_history:
    st.info("""
    🌟 **Welcome to SafeSpace**
    
    This is a safe, confidential environment where you can express your thoughts and feelings. 
    I'm here to listen, provide support, and help you find resources when needed.
    
    **Remember:** If you're experiencing a mental health crisis, please reach out to 
    professional help immediately using the resources in the sidebar.
    """)

# Quick start buttons for common topics
if not st.session_state.chat_history:
    st.markdown("### Quick Start - How are you feeling?")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_start_pressed = False
    
    with col1:
        if st.button("😰 Anxious"):
            user_input = "I'm feeling anxious and worried about everything"
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            quick_start_pressed = True
    
    with col2:
        if st.button("😔 Sad"):
            user_input = "I'm feeling really sad and down lately"
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            quick_start_pressed = True
    
    with col3:
        if st.button("😤 Stressed"):
            user_input = "I'm feeling overwhelmed and stressed out"
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            quick_start_pressed = True
    
    with col4:
        if st.button("😞 Lonely"):
            user_input = "I'm feeling lonely and disconnected"
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            quick_start_pressed = True
    
    # Process quick start button press
    if quick_start_pressed:
        st.session_state.conversation_started = True
        st.rerun()

# Chat input
user_input = st.chat_input("What's on your mind today? I'm here to listen...")

# Process user input (from text input or quick buttons)
if user_input or (st.session_state.chat_history and not st.session_state.conversation_started):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    st.session_state.conversation_started = True
    
    # Get the most recent user message
    latest_user_message = st.session_state.chat_history[-1]["content"]
    
    # Show loading indicator
    with st.spinner("Thinking and preparing a thoughtful response..."):
        response_content, tool_used, status = get_ai_response(latest_user_message)
    
    # Format the response
    if tool_used == "crisis_intervention":
        response_content = f"🚨 **Crisis Support Activated**\n\n{response_content}"
    
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response_content,
        "tool": tool_used,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    # Rerun to show the new message
    st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.markdown("### Conversation")
    
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show timestamp for assistant messages
            if msg["role"] == "assistant" and "timestamp" in msg:
                st.caption(f"Responded at {msg['timestamp']}")
    
    # Add some helpful buttons at the end of conversation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Start Fresh", help="Clear the conversation and start over"):
            st.session_state.chat_history = []
            st.session_state.conversation_started = False
            st.rerun()
    
    with col2:
        if st.button("📋 Get Resources", help="Show mental health resources"):
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": """Here are some helpful mental health resources:

**Crisis Support:**
• National Suicide Prevention Lifeline("Hello! Lifeline"): 1800-121-3667
• Crisis Text Line (Kiran Helpline): 1800-599-0019
• National Domestic Violence Hotline: 7827170170

**Mental Health Support:**
• National Alliance on Mental Illness (NAMI): 1-800-950-NAMI
• Psychology Today (Find a Therapist): https://psychologyindia.com/

**Self-Care Apps:**
• Headspace (meditation)
• Calm (sleep & relaxation)
• Mood Meter (emotion tracking)
• Sanvello (mood & anxiety tracking)

Remember: Professional help is always available when you need it.""",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
    
    with col3:
        if st.button("💙 Self-Care Tips", help="Get self-care suggestions"):
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": """Here are some gentle self-care reminders:

**Immediate Comfort:**
• Take 5 deep breaths
• Drink a glass of water
• Step outside for fresh air
• Listen to calming music

**Daily Self-Care:**
• Maintain a regular sleep schedule
• Eat nourishing meals
• Move your body gently
• Connect with supportive people

**Weekly Self-Care:**
• Engage in a hobby you enjoy
• Spend time in nature
• Practice gratitude
• Set boundaries when needed

**Remember:** Self-care isn't selfish - it's necessary. You deserve kindness, especially from yourself.""",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()

# Footer with disclaimer
st.markdown("---")
st.markdown("""
**Disclaimer:** SafeSpace is an AI support tool and is not a substitute for professional mental health treatment. 
If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately.

All conversations are processed locally and are not stored permanently.
""")