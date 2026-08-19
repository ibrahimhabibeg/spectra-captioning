import json
import argparse
import sys
import math
import statistics
from pathlib import Path

PRICES = {
    "gemini-3.7-flash": {"input": 0.75, "output": 1.50},
    "gemini-3.6-flash": {"input": 0.75, "output": 1.50},
    "gemini-3.1-pro": {"input": 2.00, "output": 4.00},
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_jsonl", type=Path)
    parser.add_argument("target_objects", type=int)
    parser.add_argument("--model", type=str, default="gemini-3.7-flash")
    parser.add_argument("--strategy", type=str, default=None, help="Filter records by strategy name")
    args = parser.parse_args()
    
    if not args.sample_jsonl.exists():
        sys.exit(1)
    if args.model not in PRICES:
        sys.exit(1)
        
    return args

def main():
    args = parse_args()

    inputs = []
    thoughts = []
    outputs = []

    with open(args.sample_jsonl, "r") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            if args.strategy and record.get("strategy") != args.strategy:
                continue
            inputs.append(record["usage"]["input_tokens"])
            thoughts.append(record["usage"].get("thought_tokens") or 0)
            outputs.append(record["usage"]["output_tokens"])

    if not inputs:
        print("Error: No records found matching the criteria.")
        sys.exit(1)

    N_sample = len(inputs)
    N_target = args.target_objects

    avg_input = statistics.mean(inputs)
    avg_thought = statistics.mean(thoughts)
    avg_output = statistics.mean(outputs)

    std_input = statistics.pstdev(inputs) if N_sample > 0 else 0
    std_thought = statistics.pstdev(thoughts) if N_sample > 0 else 0
    std_output = statistics.pstdev(outputs) if N_sample > 0 else 0

    est_total_input = N_target * avg_input
    est_total_thought = N_target * avg_thought
    est_total_output = N_target * avg_output

    std_total_input = math.sqrt(N_target) * std_input
    std_total_thought = math.sqrt(N_target) * std_thought
    std_total_output = math.sqrt(N_target) * std_output

    in_price = PRICES[args.model]["input"] / 1_000_000
    out_price = PRICES[args.model]["output"] / 1_000_000

    input_cost = est_total_input * in_price
    input_cost_std = std_total_input * in_price

    thoughts_cost = est_total_thought * out_price
    thoughts_cost_std = std_total_thought * out_price

    output_cost = est_total_output * out_price
    output_cost_std = std_total_output * out_price

    total_cost = input_cost + thoughts_cost + output_cost
    # Standard deviation of sum of independent sums is sqrt(sum of variances)
    total_cost_std = math.sqrt(input_cost_std**2 + thoughts_cost_std**2 + output_cost_std**2)

    print(f"--- Full-Scale Estimate ({N_target:,} objects) ---")
    print(f"Model: {args.model}")
    if args.strategy:
        print(f"Strategy: {args.strategy}")
    print(f"Estimated Total Input Tokens:  {est_total_input:,.0f} ± {std_total_input * 2:,.0f} (${input_cost:.2f} ± ${input_cost_std * 2:.2f})")
    print(f"Estimated Total Thought Tokens:{est_total_thought:,.0f} ± {std_total_thought * 2:,.0f} (${thoughts_cost:.2f} ± ${thoughts_cost_std * 2:.2f})")
    print(f"Estimated Total Output Tokens: {est_total_output:,.0f} ± {std_total_output * 2:,.0f} (${output_cost:.2f} ± ${output_cost_std * 2:.2f})")
    print(f"==================================================")
    print(f"TOTAL ESTIMATED COST: ${total_cost:.2f} ± ${total_cost_std * 2:.2f}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
