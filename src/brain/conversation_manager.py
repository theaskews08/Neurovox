import logging
import asyncio
import requests
import json

logger = logging.getLogger(__name__)


class ConversationalBrain:
    def __init__(self, resource_manager, memory_manager):
        self.resource_manager = resource_manager
        self.memory_manager = memory_manager
        self.current_topic = None
        self.persona = "default"
        self.conversation_state = "greeting"
        self.llm_url = "http://localhost:11434/api/generate"
        self.is_initialized = False

    async def start(self):
        """Start the conversational brain"""
        logger.info("Starting conversational brain")

        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                logger.error("Ollama API not available")
                return False

            # Check if the model is loaded
            models = response.json()
            if not any(model["name"].startswith("llama3") for model in models.get("models", [])):
                logger.warning("Llama 3 model not found in Ollama. Will be pulled on first request.")

            self.is_initialized = True
            logger.info("Conversational brain started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start conversational brain: {str(e)}")
            return False

    async def stop(self):
        """Stop the conversational brain"""
        logger.info("Stopping conversational brain")
        self.is_initialized = False
        return True

    async def health_check(self):
        """Check if the service is healthy"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False

    async def process_utterance(self, user_input):
        """Process user input with full contextual awareness"""
        if not self.is_initialized:
            logger.error("Cannot process utterance: Brain not initialized")
            return "I'm sorry, but I'm not fully initialized yet. Please try again in a moment."

        try:
            # Get conversation context
            context = await self.memory_manager.get_recent_context()

            # Generate system prompt based on current state
            system_prompt = self._generate_system_prompt()

            # Create messages array for the API
            messages = []

            # Add system message
            messages.append({"role": "system", "content": system_prompt})

            # Add conversation history
            for msg in context:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Add current user input
            messages.append({"role": "user", "content": user_input})

            # Call Ollama API
            response = requests.post(
                self.llm_url,
                json={
                    "model": "llama3:8b-q4_0",
                    "prompt": user_input,
                    "system": system_prompt,
                    "stream": False,
                    "context": self._get_context_for_ollama(context)
                }
            )

            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.text}")
                return "I'm sorry, but I encountered an error. Please try again."

            response_json = response.json()
            assistant_response = response_json.get("response", "")

            # Update conversation memory
            await self.memory_manager.add_exchange(user_input, assistant_response)

            # Update conversation state
            self._update_conversation_state(user_input, assistant_response)

            return assistant_response

        except Exception as e:
            logger.error(f"Error processing utterance: {str(e)}")
            return "I'm sorry, but I encountered an error. Please try again."

    def _generate_system_prompt(self):
        """Generate appropriate system prompt based on current state"""
        base_prompt = "You are a helpful voice assistant named NeuroVox."

        state_prompts = {
            "greeting": "Provide a warm, brief greeting and offer assistance.",
            "topic_transition": "Acknowledge the topic change and respond appropriately.",
            "deep_conversation": "Provide thoughtful, nuanced responses while maintaining conversational flow.",
            "task_oriented": "Focus on clear, actionable information to help complete the task.",
            "emotional_support": "Show empathy and provide supportive responses."
        }

        return f"{base_prompt} {state_prompts.get(self.conversation_state, '')} Keep your responses concise since they will be spoken aloud."

    def _update_conversation_state(self, user_input, response):
        """Update conversation state based on exchange"""
        # Use simple keyword detection for state transitions
        if "how are you" in user_input.lower() or "feeling" in user_input.lower():
            self.conversation_state = "emotional_support"
        elif any(task in user_input.lower() for task in ["set", "remind", "calculate", "what is", "how do"]):
            self.conversation_state = "task_oriented"
        elif len(user_input.split()) > 15:  # Longer utterances suggest deeper conversation
            self.conversation_state = "deep_conversation"

    def _get_context_for_ollama(self, context):
        """Convert context to format expected by Ollama API"""
        # Ollama expects a list of strings for context
        if not context:
            return []

        context_str = []
        for msg in context:
            if msg["role"] == "user":
                context_str.append(f"Human: {msg['content']}")
            else:
                context_str.append(f"Assistant: {msg['content']}")

        return context_str