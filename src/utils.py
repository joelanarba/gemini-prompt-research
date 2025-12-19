from openai import OpenAI
from datetime import datetime
import csv
import os
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

# Initialize OpenRouter client
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

def load_prompt(filename):
    """Load system prompt from file."""
    filepath = os.path.join("prompts", filename)
    if not os.path.exists(filepath):
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_queries(filepath):
    """Load test queries from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def call_openrouter(model_name, system_prompt, user_query, temperature=0.7):
    """
    Call OpenRouter API with system prompt and user query.
    Returns the response text and metadata.
    """
    try:
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Make API call
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        
        return {
            "response": response.choices[0].message.content,
            "success": True,
            "error": None,
            "model": response.model,
            "tokens_used": response.usage.total_tokens if response.usage else None
        }
        
    except Exception as e:
        return {
            "response": "",
            "success": False,
            "error": str(e),
            "model": model_name,
            "tokens_used": None
        }

def log_response(query, prompt_type, trial, response_data, output_dir):
    """Log individual response to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"responses_{timestamp}.csv")
    
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "timestamp", "query", "prompt_type", "trial", 
                "response", "success", "error", "model", "tokens_used"
            ])
        
        writer.writerow([
            datetime.now().isoformat(),
            query,
            prompt_type,
            trial,
            response_data["response"],
            response_data["success"],
            response_data["error"],
            response_data.get("model", ""),
            response_data.get("tokens_used", "")
        ])
    
    return filename