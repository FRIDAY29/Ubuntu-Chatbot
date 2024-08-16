# Ubuntu Chatbot

Welcome to the Ubuntu Chatbot project! This repository contains a chatbot designed to operate on Ubuntu systems, capable of understanding and executing basic commands.

## Overview

The Ubuntu Chatbot is a command-line based chatbot that can perform various system tasks and respond to user queries. It's built using Python and designed to be easily extendable for additional functionalities.

## Features

- **Natural Language Understanding**: Processes user input and executes commands based on the content.
- **System Interaction**: Can run system commands and scripts, such as listing files and checking the current time.
- **Customizable**: Easily extendable to add more commands and responses.

## Installation

Follow these steps to install and run the Ubuntu Chatbot on your local machine:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/FRIDAY29/Ubuntu-Chatbot.git
   cd Ubuntu-Chatbot
pip install -r requirements.txt
python chatbot.py

### `chatbot.py`

```python
import subprocess
import datetime

def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def list_files():
    try:
        files = subprocess.check_output(['ls', '-l']).decode('utf-8')
        return files
    except Exception as e:
        return str(e)

def process_command(command):
    command = command.lower()
    if 'time' in command:
        return f"Current time is: {get_current_time()}"
    elif 'list' in command or 'files' in command:
        return list_files()
    elif 'exit' in command:
        return "Goodbye!"
    else:
        return "Sorry, I didn't understand that command."

def main():
    print("Ubuntu Chatbot is running. Type 'exit' to end the session.")
    while True:
        user_input = input("You: ")
        response = process_command(user_input)
        print(f"Chatbot: {response}")
        if 'exit' in user_input.lower():
            break

if __name__ == "__main__":
    main()
# No additional libraries required for this basic chatbot
# config.py

# Example configuration (not used in this basic example)
