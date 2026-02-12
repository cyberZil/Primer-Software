#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import re
import whisper_prompt
import wake_word
from render_avatar import erase_area, set_stage, render_paragraph, render_word
# --- IMPORT FROM OLLAMA-CHAT.PY ---
from ollama_chat import get_primer_response, INITIAL_MESSAGES_HISTORY 

# --- Configuration ---
TRIGGER_WORD = "Primer"
OLLAMA_MODEL = "gemma3:1b" 
RECORD_FILE = "recording.wav"

# Initialize the conversation history using the imported setup
messages_history = INITIAL_MESSAGES_HISTORY

# --- Core Logic Functions ---

def parse_and_display(ai_response: str):
    """
    Parses the mood, sets the avatar, and renders the text on the e-ink display.
    """
    moods = ["Neutral", "Laughing", "Confused", "Celebratory", "Sad", "Sleeping"]
    
    # Use regex to find the mood word at the start, case-insensitive
    match = re.match(r"(\w+): ", ai_response, re.IGNORECASE)
    
    if match and match.group(1).capitalize() in moods:
        emotion = match.group(1).capitalize()
        text = ai_response[len(match.group(0)):].strip()
    else:
        # Default to Neutral if parsing fails
        emotion = "Neutral"
        text = ai_response
    
    print(f"Parsed Emotion: {emotion}")
    print(f"Text to Display: {text}")

    # Set the avatar
    set_stage(emotion)
    time.sleep(1) # Wait for initial clear/draw to complete

    # Render the text
    render_paragraph(text)
    time.sleep(5) # Keep text on screen for a moment

def main_loop():
    """
    The main loop of the AI teacher application.
    """
    global messages_history # Need to update the global history list
    messages_history = INITIAL_MESSAGES_HISTORY
    
    try:
        # Initial display state
        set_stage("Neutral")
        #render_paragraph(f"Hello, I am Primer. I am listening for my name, '{TRIGGER_WORD}'  Say 'Bye' to end our lesson.")
        render_paragraph(f"Hello, I am Primer. Start each question with my name. Say ' Primer Bye' when you're done.")
        runner = whisper_prompt.WhisperONNXRunner()
        render_word("Listening...")
        
        while True:   
            
            wakeWord = wake_word.wake_word()

            if wakeWord:
                print(f"\n✅ Wake word '{TRIGGER_WORD}' detected!")
                
                user_question = whisper_prompt.get_question(runner, device=None)
                
                # Handle None return from get_question
                if user_question is None:
                    print("❌ No speech detected. Re-listening.")
                    set_stage("Confused")
                    render_paragraph(f"Can you try saying my name again? I didnt catch that.")
                    render_word("Listening...")
                    continue
                
                # Remove trigger word from question
                user_question = re.sub(TRIGGER_WORD, '', user_question, flags=re.IGNORECASE).strip()
                
                # Check for blank audio or silence
                if not user_question or re.search(r'^\s*(\[?blank_audio\]?|[\[\(]?(blank|silence|quiet|sound)[\]\)]?)\s*$', user_question, re.IGNORECASE):
                    print("❌ Blank audio or no speech detected. Re-listening.")
                    set_stage("Confused")
                    render_paragraph(f"Can you try saying my name again? I didnt catch that.")
                    render_word("Listening...")
                    continue
                    
                print(f"\nSending user question to Ollama: {user_question}")
                # 3. Get AI Response using the imported function
                erase_area(20, 700, 200, 730)
                render_word("Thinking...")
                ai_response, messages_history = get_primer_response( user_question, messages_history, OLLAMA_MODEL)
                erase_area(20, 700, 250, 730)
                print(f"\n🧠 AI Response: {ai_response}")
                if "bye" in user_question.lower():
                    print("Received exit command from AI. Exiting main loop.")
                    parse_and_display(ai_response)
                    break

                # 4. Parse and Display
                print("Parsing and displaying AI response...")
                parse_and_display(ai_response + '   Can I help with something else? Say "Primer Bye" to end our lesson.')
            else:
                set_stage("Confused")
                render_paragraph(f"Can you try saying my name again? I didnt catch that.")
                print(f"\n❌ Trigger word '{TRIGGER_WORD}' NOT detected in transcription.")         
            
            render_word("Listening...")

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        # Clean up the display and audio resources
        set_stage("Sleeping") # Clears display and puts it to sleep
        erase_area(20, 700, 250, 730)
        print("Primer is signing off.")

if __name__ == "__main__":
    main_loop()

###### END OF FILE ######