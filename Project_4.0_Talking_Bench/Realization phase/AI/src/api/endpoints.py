from flask import Flask, request, jsonify
import sys
import os
import asyncio
from typing import Dict, Any
import logging
import uuid
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.conversation_manager import ConversationManager
from modules.emotion_analyzer import EmotionAnalyzer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active and completed conversations
active_conversations: Dict[str, ConversationManager] = {}
completed_conversations: Dict[str, ConversationManager] = {}
emotion_analyzer = EmotionAnalyzer()

def run_async(coro):
    """Helper function to run async code in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the AI Service API"}), 200

@app.route('/start-conversation', methods=['POST'])
def start_conversation():
    try:
        # Generate a temporary session token
        session_token = str(uuid.uuid4())
        
        # Create conversation manager
        conversation_manager = run_async(ConversationManager.create())
        active_conversations[session_token] = conversation_manager
        
        welcome_message = conversation_manager.get_welcome_message()
        current_question = conversation_manager.get_current_question()
        
        return jsonify({
            "welcome_message": welcome_message,
            "current_question": current_question,
            "session_token": session_token
        })
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/process-message', methods=['POST'])
def process_message():
    try:
        data = request.json
        session_token = data.get('session_token')
        user_input = data.get('message')
        
        if not session_token or not user_input:
            return jsonify({"error": "Missing required fields"}), 400
            
        if session_token not in active_conversations:
            return jsonify({"error": "Session not found"}), 404
            
        conversation_manager = active_conversations[session_token]
        response = run_async(conversation_manager.process_conversation(user_input))
        
        # Check if conversation has ended
        if response.get('metadata', {}).get('conversation_ended', False):
            # Get final analytics
            final_analytics = conversation_manager.get_conversation_analytics()
            emotional_analysis = run_async(emotion_analyzer.get_conversation_insights(
                conversation_manager.conversation_history
            ))
            
            # Move conversation to completed conversations
            completed_conversations[session_token] = active_conversations[session_token]
            del active_conversations[session_token]
            
            return jsonify({
                "response": response['response'],
                "conversation_ended": True,
                "analytics": final_analytics,
                "emotional_analysis": emotional_analysis
            })
            
        return jsonify({
            "response": response['response'],
            "follow_ups": response.get('follow_ups', []),
            "metadata": response.get('metadata', {}),
            "emotional_analysis": response.get('emotional_analysis', {})
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-conversation-state', methods=['GET'])
def get_conversation_state():
    session_token = request.args.get('session_token')
    
    if not session_token:
        return jsonify({"error": "Missing session_token"}), 400
        
    # Check both active and completed conversations
    if session_token in active_conversations:
        conversation_manager = active_conversations[session_token]
    elif session_token in completed_conversations:
        conversation_manager = completed_conversations[session_token]
    else:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify({
        "current_question": conversation_manager.get_current_question(),
        "summary": conversation_manager.get_conversation_summary(),
        "follow_up_count": conversation_manager.follow_up_count,
        "max_follow_ups": conversation_manager.max_follow_ups,
        "question_index": conversation_manager.current_question_index,
        "total_questions": len(conversation_manager.questions)
    })

@app.route('/end-conversation', methods=['POST'])
def end_conversation():
    try:
        data = request.json
        session_token = data.get('session_token')
        
        if not session_token:
            return jsonify({"error": "Missing session_token"}), 400
            
        if session_token not in active_conversations:
            return jsonify({"error": "Session not found"}), 404
            
        conversation_manager = active_conversations[session_token]
        final_response = run_async(conversation_manager.end_conversation())
        
        # Get final emotional analysis
        emotional_analysis = run_async(emotion_analyzer.get_conversation_insights(
            conversation_manager.conversation_history
        ))
        
        # Move to completed conversations instead of deleting
        completed_conversations[session_token] = active_conversations[session_token]
        del active_conversations[session_token]
        
        return jsonify({
            "response": final_response['response'],
            "analytics": final_response['analytics'],
            "emotional_analysis": emotional_analysis,
            "metadata": final_response['metadata']
        })
    except Exception as e:
        logger.error(f"Error ending conversation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-emotional-analysis', methods=['GET'])
def get_emotional_analysis():
    session_token = request.args.get('session_token')
    
    if not session_token:
        return jsonify({"error": "Missing session_token"}), 400
    
    # Check both active and completed conversations
    if session_token in active_conversations:
        conversation_manager = active_conversations[session_token]
    elif session_token in completed_conversations:
        conversation_manager = completed_conversations[session_token]
    else:
        return jsonify({"error": "Session not found"}), 404
    
    emotional_analysis = run_async(emotion_analyzer.get_conversation_insights(
        conversation_manager.conversation_history
    ))
    
    return jsonify(emotional_analysis)

if __name__ == '__main__':
    try:
        print("Starting server on port 8080...")
        app.run(
            host='0.0.0.0',
            port=8080,
            debug=False  # Disable debug mode
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("Please make sure port 8080 is available")