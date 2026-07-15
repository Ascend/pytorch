import json
import os
import argparse

def convert_alpaca_to_baichuan(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted_data = []

    for item in data:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()

        if input_text:
            human_text = instruction + "\n" + input_text
        else:
            human_text = instruction

        new_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": human_text
                },
                {
                    "from": "assistant",
                    "value": output_text
                }
            ]
        }

        converted_data.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"Conversion completed, total {len(converted_data)} records")
    print(f"Output file: {output_path}")

def build_argparser():
    p = argparse.ArgumentParser(description="Convert Alpaca JSON to Baichuan conversations JSON")
    p.add_argument("--input", "-i", required=True, help="Path to alpaca json file")
    p.add_argument("--output", "-o", required=True, help="Path to output json file")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    convert_alpaca_to_baichuan(args.input, args.output)