from config import *
from utils import load_prompt, load_queries, call_openrouter, log_response
import time

def run_experiment():
    """
    Main experiment runner.
    Tests each query with each prompt type across multiple trials.
    """
    
    print("=" * 60)
    print("OPENROUTER GEMINI PROMPT EVALUATION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE}")
    print()
    
    # Load prompts
    prompts = {
        "baseline": load_prompt("baseline.txt"),
        "structured": load_prompt("structured.txt"),
        "empathetic": load_prompt("empathetic.txt")
    }
    
    # Load test queries
    queries = load_queries(QUERIES_FILE)
    
    print(f"Starting experiment with {len(queries)} queries")
    print(f"Running {NUM_TRIALS} trials per prompt type")
    print(f"Total API calls: {len(queries) * len(prompts) * NUM_TRIALS}")
    print("-" * 60)
    
    total_cost_estimate = len(queries) * len(prompts) * NUM_TRIALS * 0.000075
    print(f"Estimated cost: ${total_cost_estimate:.4f}")
    print("-" * 60)
    print()
    
    # Run experiment
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}/{len(queries)}: {query[:50]}...")
        
        for prompt_name, prompt_text in prompts.items():
            print(f"  Testing with '{prompt_name}' prompt...")
            
            for trial in range(1, NUM_TRIALS + 1):
                print(f"    Trial {trial}/{NUM_TRIALS}...", end=" ")
                
                # Call API
                response_data = call_openrouter(
                    MODEL_NAME,
                    prompt_text, 
                    query, 
                    temperature=TEMPERATURE
                )
                
                # Log response
                output_file = log_response(
                    query, 
                    prompt_name, 
                    trial, 
                    response_data, 
                    OUTPUT_DIR
                )
                
                if response_data["success"]:
                    tokens = response_data.get("tokens_used", "?")
                    print(f"✓ ({tokens} tokens)")
                else:
                    print(f"✗ ({response_data['error']})")
                
                # Rate limiting: wait between calls
                time.sleep(1)
    
    print("\n" + "=" * 60)
    print(f"Experiment complete! Results saved to {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()


"""

## **Available Models on OpenRouter**

#You can use any of these models (just change `MODEL_NAME` in `.env`):

### **Gemini Models (Google):**

google/gemini-flash-1.5          # Fastest, cheapest (~$0.000075/request)
google/gemini-pro-1.5            # More capable (~$0.001/request)
google/gemini-2.0-flash-exp:free # FREE experimental version


### **Other Models (for comparison):**
'''```
anthropic/claude-3.5-sonnet      # Claude (best quality)
#openai/gpt-4o-mini               # GPT-4 (fast and cheap)
meta-llama/llama-3.1-8b-instruct # Open source
'''

"""