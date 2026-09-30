import json
import os
import time
import glob
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

def process_chunk(chunk_path, client):
    output_path = chunk_path.replace('.json', '_labeled.json')
    if os.path.exists(output_path):
        return f"Skipped {chunk_path}, already processed."

    with open(chunk_path, 'r') as f:
        data = json.load(f)

    prompt = f"""
    You are a clinical pharmacologist. Assess the severity of these {len(data)} drug-drug interaction descriptions.
    Return a JSON object with a 'labels' array mapping the id to the severity.
    Severities MUST be exactly one of: "Mild", "Moderate", "Severe".
    If it says "fatal" or "contraindicated", it's "Severe".
    If it says "may increase", it's "Moderate".
    If it says "minor" or "limited data", it's "Mild".

    Respond ONLY with valid JSON matching this schema:
    {{
      "labels": [
        {{"id": 123, "severity": "Moderate"}},
        ...
      ]
    }}

    Input items:
    {json.dumps(data)}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )

        result_json = response.text
        # Strip potential markdown blocks
        if result_json.startswith('```json'):
            result_json = result_json[7:]
        if result_json.endswith('```'):
            result_json = result_json[:-3]

        parsed = json.loads(result_json.strip())

        with open(output_path, 'w') as f:
            json.dump(parsed['labels'], f, indent=2)

        return f"Success: {chunk_path}"
    except Exception as e:
        return f"Error on {chunk_path}: {str(e)}"

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment.")
        return

    client = genai.Client(api_key=api_key)

    chunk_files = sorted(glob.glob('chunks/chunk_*.json'))
    # Filter out labeled files
    chunk_files = [f for f in chunk_files if not f.endswith('_labeled.json')]

    print(f"Found {len(chunk_files)} chunks to process.")

    # Run 5 threads concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_chunk, f, client): f for f in chunk_files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"[{i+1}/{len(chunk_files)}] {result}")
            # Add small sleep to prevent typical rate limits
            time.sleep(1)

if __name__ == "__main__":
    main()