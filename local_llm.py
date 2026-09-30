import base64
import json
import urllib.request
import urllib.error
from typing import List, Optional

def query_local_llm(
    prompt: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    model: str = "ministral-3:8b"
) -> str:
    """
    Sends a text prompt and optional images to a locally running Ollama LLM.
    
    Args:
        prompt: Optional text prompt to send to the model.
        image_paths: Optional list of file paths to images.
        model: The name of the Ollama model to use.
        
    Returns:
        The response string from the LLM.
    """
    url = "http://localhost:11434/api/chat"
    
    # Ensure there is at least some content
    message_content = prompt if prompt else ""
    
    message = {
        "role": "user",
        "content": message_content
    }
    
    if image_paths:
        base64_images = []
        for path in image_paths:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                base64_images.append(encoded_string)
        message["images"] = base64_images

    payload = {
        "model": model,
        "messages": [message],
        "stream": False
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("message", {}).get("content", "")
    except urllib.error.URLError as e:
        return f"Error communicating with local LLM. Is Ollama running on localhost:11434 and the {model} model installed? Details: {e}"
