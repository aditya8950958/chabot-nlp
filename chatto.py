import random
import json
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import re

class ExactAnswerChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        self.load_intents()
        self.setup_answer_generators()
        
    def setup_answer_generators(self):
        """Setup exact answer generation based on question analysis"""
        # Expanded knowledge base with more topics
        self.knowledge_base = {
            "programming": {
                "python": "Python is a high-level, interpreted programming language known for its simple syntax and versatility.",
                "javascript": "JavaScript is a programming language primarily used for web development to create interactive web pages.",
                "machine_learning": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
                "coding": "Coding is the process of creating instructions for computers using programming languages.",
                "programming": "Programming is the process of designing and building executable computer software to accomplish specific tasks."
            },
            "networking": {
                "tcp": "TCP/IP is a suite of communication protocols used to interconnect network devices on the internet.",
                "ip": "TCP/IP is a suite of communication protocols used to interconnect network devices on the internet.",
                "router": "A router is a networking device that forwards data packets between computer networks.",
                "firewall": "A firewall is a network security system that monitors and controls incoming and outgoing network traffic.",
                "network": "A network is a collection of interconnected devices that can communicate and share resources."
            },
            "general": {
                "chatbot": "A chatbot is a computer program designed to simulate conversation with human users through text or voice interactions.",
                "ai": "Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think and learn.",
                "life": "Life is the condition that distinguishes animals and plants from inorganic matter, including the capacity for growth, reproduction, functional activity, and continual change.",
                "privacy": "Privacy is the right of individuals to control access to their personal information and activities.",
                "hello": "Hello is a greeting used to acknowledge someone's presence or to begin a conversation.",
                "computer": "A computer is an electronic device that processes data according to programmed instructions.",
                "internet": "The internet is a global network of interconnected computers that communicate using standardized protocols."
            },
            "concepts": {
                "love": "Love is a complex emotion characterized by deep affection, care, and attachment to someone or something.",
                "happiness": "Happiness is a positive emotional state characterized by feelings of joy, satisfaction, and fulfillment.",
                "success": "Success is the accomplishment of goals or the attainment of desired outcomes through effort and skill.",
                "learning": "Learning is the process of acquiring new knowledge, skills, behaviors, or understanding through experience or study."
            }
        }

    def extract_topic_keywords(self, user_input):
        """Extract main topic from user question - FIXED VERSION"""
        # Remove question words and common words
        stop_words = ["what", "is", "how", "to", "do", "i", "can", "why", "does", 
                     "when", "should", "where", "which", "the", "a", "an", "and", "or", "?"]
        
        # Clean and split input
        cleaned_input = re.sub(r'[^\w\s]', '', user_input.lower())
        words = cleaned_input.split()
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords

    def find_exact_answer(self, keywords):
        """Find exact answer from knowledge base - FIXED VERSION"""
        if not keywords:
            return None, None, None
            
        # Search through knowledge base
        for category, topics in self.knowledge_base.items():
            for topic, answer in topics.items():
                # Check if any keyword matches the topic
                for keyword in keywords:
                    if keyword.lower() == topic.lower() or keyword.lower() in topic.lower():
                        return answer, topic, category
                    # Also check if topic is in keyword (for compound words)
                    if topic.lower() in keyword.lower():
                        return answer, topic, category
        
        return None, None, None

    def analyze_question_type(self, user_input):
        """Analyze question to determine exact answer type needed"""
        user_lower = user_input.lower()
        
        if any(phrase in user_lower for phrase in ["what is", "what's", "define", "explain what"]):
            return "definition"
        elif any(phrase in user_lower for phrase in ["how to", "how do i", "how can i", "steps to"]):
            return "process"
        elif any(phrase in user_lower for phrase in ["why does", "why is", "reason for", "because"]):
            return "explanation"
        elif any(phrase in user_lower for phrase in ["when should", "when to", "what time"]):
            return "timing"
        elif any(phrase in user_lower for phrase in ["where can", "where to", "location"]):
            return "location"
        else:
            return "general"

    def get_exact_response(self, user_input):
        """Generate exact answer based on user question - FIXED VERSION"""
        # Handle simple greetings first
        simple_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if user_input.lower().strip() in simple_greetings:
            return self.get_base_response("greeting")
        
        # Handle thanks
        if any(word in user_input.lower() for word in ["thank", "thanks"]):
            return self.get_base_response("thanks")
        
        # Handle goodbye
        if any(word in user_input.lower() for word in ["bye", "goodbye", "see you"]):
            return self.get_base_response("goodbye")
        
        # Analyze question type
        question_type = self.analyze_question_type(user_input)
        
        # Extract keywords
        keywords = self.extract_topic_keywords(user_input)
        
        # Find exact answer
        exact_answer, topic, category = self.find_exact_answer(keywords)
        
        # Generate response based on findings
        if exact_answer:
            if question_type == "definition":
                return exact_answer
            elif question_type == "process":
                if topic in ["programming", "coding"]:
                    return f"{exact_answer} To get started: 1) Choose a language, 2) Set up your environment, 3) Learn basics, 4) Practice with projects."
                else:
                    return f"{exact_answer} Let me know if you need specific steps for this topic."
            elif question_type == "explanation":
                return f"{exact_answer} This is important because it forms the foundation for understanding related concepts."
            else:
                return exact_answer
        else:
            # Fallback to intent classification for unknown topics
            intent_tag, confidence = self.classify_intent(user_input)
            if confidence > 0.7:
                return self.get_base_response(intent_tag)
            else:
                # Better fallback response
                if keywords:
                    return f"I don't have specific information about '{' '.join(keywords)}' in my knowledge base. Could you ask about a different topic or provide more context?"
                else:
                    return "I'm not sure what you're asking about. Could you rephrase your question or be more specific?"

    # Include your existing methods
    def load_model(self):
        """Load the trained model and data"""
        try:
            FILE = "data.pth"
            data = torch.load(FILE)
            
            self.input_size = data["input_size"]
            self.hidden_size = data["hidden_size"]
            self.output_size = data["output_size"]
            self.all_words = data['all_words']
            self.tags = data['tags']
            model_state = data["model_state"]
            
            self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_intents(self):
        """Load intents from JSON file"""
        try:
            with open('intents.json', 'r') as json_data:
                self.intents = json.load(json_data)
        except Exception as e:
            print(f"Error loading intents: {e}")
            raise
    
    def classify_intent(self, message):
        """Classify intent using the trained model"""
        sentence = tokenize(message)
        X = bag_of_words(sentence, self.all_words)
        X = np.array(X).reshape(1, -1)
        X = torch.from_numpy(X).to(self.device).float()
        
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        
        tag = self.tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        return tag, prob.item()
    
    def get_base_response(self, intent_tag):
        """Get base response from intents"""
        for intent in self.intents['intents']:
            if intent_tag == intent["tag"]:
                return random.choice(intent['responses'])
        return "I'm not sure how to respond to that."

# Usage
if __name__ == "__main__":
    try:
        chatbot = ExactAnswerChatbot()
        
        print("Fixed Exact Answer Chatbot is ready! (type 'quit' to exit)")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                break
            
            response = chatbot.get_exact_response(user_input)
            print(f"Bot: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
