#!/usr/bin/env python3
import os
import requests
import subprocess
import json
import re
import argparse
import base64  # For image encoding
from typing import List, Dict

# Counter for automated filename generation
file_counter = 1

def call_ollama_api_stream(messages: List[Dict], model: str = "gemma3:latest", max_retries: int = 3) -> str:
    """Calls the Ollama API with retries, streaming the response."""
    data = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    url = "http://localhost:11434/api/chat"
    
    for attempt in range(max_retries):
        try:
            with requests.post(url, json=data, stream=True, timeout=30) as response:
                response.raise_for_status()
                full_response = ""
                for chunk in response.iter_lines():
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8')
                        json_chunk = json.loads(decoded_chunk)
                        content = json_chunk.get("message", {}).get("content", "")
                        full_response += content
                        print(content, end="", flush=True)
                return full_response
        except requests.exceptions.RequestException as e:
            print(f"\nError calling Ollama API (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None

def encode_image(image_path: str) -> str:
    """Encodes an image to base64 for vision models."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def extract_code_block(response_text: str) -> tuple:
    """Extracts a code block, its language, and an optional filename from the AI's response."""
    filename_match = re.search(r"filename: (.*?)\n", response_text)
    filename = filename_match.group(1).strip() if filename_match else None

    code_match = re.search(r"```(.*?)\n(.*?)```", response_text, re.DOTALL)
    if code_match:
        language = code_match.group(1).strip().lower() or "python"  # Default to Python
        code = code_match.group(2).strip()
        return filename, language, code
    return filename, None, None

def save_history(messages: List[Dict], file_path: str = "chat_history.json"):
    """Saves chat history to a file."""
    try:
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=4)
        print(f"\nChat history saved to {file_path}")
    except Exception as e:
        print(f"Error saving history: {e}")

def load_history(file_path: str = "chat_history.json") -> List[Dict]:
    """Loads chat history from a file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Gemini-like CLI for Ollama: Interactive AI chat with Python-focused code execution.")
    parser.add_argument("--model", default="gemma3:latest", help="Ollama model to use (e.g., gemma3:latest, llava for vision).")
    parser.add_argument("--prompt", help="One-shot prompt to send (non-interactive).")
    parser.add_argument("--image", help="Path to an image file for vision models.")
    parser.add_argument("--load-history", action="store_true", help="Load previous chat history.")
    parser.add_argument("--save-history", action="store_true", help="Save chat history on exit.")
    args = parser.parse_args()

    print(f"\nWelcome to Ollama CLI (Gemini-inspired)! Using model: {args.model}")
    print("Type 'exit' or 'quit' to end. Use '/save' to save history, '/load' to load.")

    messages = [{
        "role": "system",
        "content": "You are a helpful assistant focused on Python programming. Respond concisely. Prefer Python for code solutions. For shell commands, use ```bash blocks. For code, specify 'filename: path/to/file.py' before ```python blocks."
    }]

    if args.load_history:
        loaded = load_history()
        if loaded:
            messages.extend(loaded)
            print("Loaded previous chat history.")

    if args.prompt:  # One-shot mode
        user_content = args.prompt
        if args.image:
            base64_image = encode_image(args.image)
            if base64_image:
                user_content += f"\n[Image attached: Analyze this image.]"
                messages.append({"role": "user", "content": user_content, "images": [base64_image]})
            else:
                messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_content})
        
        print("\nAI: ", end="")
        ai_response = call_ollama_api_stream(messages, model=args.model)
        if ai_response:
            handle_response(ai_response, messages, args.model)
        return

    # Interactive mode
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            if args.save_history:
                save_history(messages)
            break
        elif prompt == "/save":
            save_history(messages)
            continue
        elif prompt == "/load":
            loaded = load_history()
            if loaded:
                messages.extend(loaded)
                print("Chat history loaded.")
            continue

        user_content = prompt
        if args.image:  # Image support in interactive mode via flag
            base64_image = encode_image(args.image)
            if base64_image:
                user_content += "\n[Image attached.]"
                messages.append({"role": "user", "content": user_content, "images": [base64_image]})
                args.image = None  # Clear after use
            else:
                messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_content})

        print("\nAI: ", end="")
        ai_response = call_ollama_api_stream(messages, model=args.model)
        if ai_response:
            messages.append({"role": "assistant", "content": ai_response})
            handle_response(ai_response, messages, args.model)

def handle_response(ai_response: str, messages: List[Dict], model: str):
    """Handles code extraction, execution, and debugging, with Python focus."""
    global file_counter
    filename, language, code = extract_code_block(ai_response)
    if not code:
        return

    # Default to Python if language is not specified
    if not language:
        language = "python"

    # Automated filename if not provided
    if not filename:
        while True:
            filename = f"generated_py_script_{file_counter}.py"
            if not os.path.exists(filename):
                break
            file_counter += 1

    # Save code
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(code)
    print(f"\nCode saved to {filename}")

    # Run confirmation
    run_confirm = input(f"\nRun this {language} code? [y/N]: ").strip().lower()
    if run_confirm != 'y':
        return

    # Generate run command via AI (Python-focused default)
    run_messages = [
        {"role": "system", "content": "Provide a shell command to run the given file, preferring Python execution (e.g., python3 file.py)."},
        {"role": "user", "content": f"Command to run '{filename}' in {language}?"}
    ]
    print("\nGenerating run command: ", end="")
    run_command_response = call_ollama_api_stream(run_messages, model=model)
    if not run_command_response:
        return

    _, _, run_command = extract_code_block(run_command_response)
    if not run_command:
        print("\nNo valid run command generated.")
        return

    # Execution with debugging loop
    while True:
        confirm = input(f"\nExecute: '{run_command}'? [y/N]: ").strip().lower()
        if confirm != 'y':
            break
        try:
            result = subprocess.run(run_command, shell=True, check=True, capture_output=True, text=True)
            print("\nSuccess! Output:\n", result.stdout)
            break
        except subprocess.CalledProcessError as e:
            print(f"\nError: {e}\nStderr:\n{e.stderr}")
            debug_confirm = input("\nDebug this error? [y/N]: ").strip().lower()
            if debug_confirm != 'y':
                break
            debug_messages = messages + [{"role": "user", "content": f"Command failed: {e.stderr}\nProvide corrected command."}]
            print("\nDebugging: ", end="")
            debug_response = call_ollama_api_stream(debug_messages, model=model)
            if debug_response:
                messages.append({"role": "assistant", "content": debug_response})
                _, _, run_command = extract_code_block(debug_response)
                if not run_command:
                    print("\nNo fix provided.")
                    break

if __name__ == "__main__":
    main()
