import time
from src.loader import load_data
from src.metrics import LLMEvaluator

def main():
    print("--- Starting LLM Evaluation Pipeline (CLI) ---")
    
    # 1. Define Paths
    chat_file = "data/chat_history.json"
    ctx_file = "data/context_data.json"
    
    # 2. Load Data
    print(f"Loading data from {chat_file}...")
    dataset = load_data(chat_file, ctx_file)
    
    if not dataset:
        print("Error: No data found.")
        return

    # 3. Initialize Engine
    evaluator = LLMEvaluator()
    
    # 4. Run Batch Evaluation
    print(f"\nEvaluating {len(dataset)} test case(s)...\n")
    
    for i, data in enumerate(dataset):
        print(f"--- Case {i+1} ---")
        print(f"Query: {data['query']}")
        
        start = time.time()
        
        # Calculate Metrics
        rel_score = evaluator.calc_relevance(data['query'], data['response'])
        faith_score = evaluator.calc_faithfulness(data['context'], data['response'])
        
        latency = (time.time() - start) * 1000
        
        # Output Report
        print(f"Relevance:    {rel_score:.4f}")
        print(f"Faithfulness: {faith_score:.4f}")
        print(f"Latency:      {latency:.2f} ms")
        
        if faith_score < 0.5:
            print("Status:       FAIL (Hallucination Detected)")
        else:
            print("Status:       PASS")
            
        print("-" * 30)

if __name__ == "__main__":
    main()