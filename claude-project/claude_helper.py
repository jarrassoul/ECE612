import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

def ask_claude(prompt):
    # Initialize the Anthropic client with your API key
    client = anthropic.Client(api_key=os.getenv('CLAUDE_API_KEY'))
    
    try:
        # Create a message and get the response
        message = client.messages.create(
            model="claude-3-opus-20240229",  # Updated to latest available model
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"