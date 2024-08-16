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
