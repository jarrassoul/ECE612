# Create a new file test_claude.py
from claude_helper import ask_claude

# Test the connection
response = ask_claude("Hello, Claude! Can you hear me?")
print(response)