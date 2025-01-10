import openai
import os
import json
import argparse
import random


def generate_scenario(basic_scenario, target_emotion, names, num_samples=3):

    synthetic_data = [] 

    # Prompt template with instruction
    prompt = f"""
    The following is a factual description of a social interaction based on the situation: "{basic_scenario}". 
    During the interaction, an event occurs that causes one person to feel {target_emotion}. This emotion should be displayed primarily through their actions and behavior, rather than explicitly stated. The cause of the emotion should be implied by the event, and the description should focus on what an external observer could see and hear.

    The description should be objective, simple, and neutral, as if it is part of a psychological study. Use clear and direct language, avoiding any story-telling or creative narrative styles. Focus only on what an external observer could see and hear, including actions, behaviors, gestures, and expressions that hint at both the emotion and its cause. Do not include internal thoughts, background information, or any unnecessary details. 
    """

    # Replace placeholder names with two random names
    sampled_names = random.sample(names, k=2)
    prompt = prompt.replace("PersonX", sampled_names[0]).replace("PersonY", sampled_names[1])
    
    # Insert prompt into chat template
    messages = [
        {"role": "system", "content": "You are an assistant that generates simple, factual social interaction scenarios for psychological research."},
        {"role": "user", "content": prompt}
    ]

    # Generate response with OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=200,
        temperature=0.6,
        n=num_samples,
    )
    
    synthetic_data.append({f"choice {i}": c.message.content for i, c in enumerate(response.choices)})
    
    return synthetic_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--basic_scenarios", type=str, default="basic_scenarios.json")
    parser.add_argument("--names", type=str, default="common_names.json")
    args = parser.parse_args()

    # Insert OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Load basic scenarios from file
    basic_scenarios = json.load(open(args.basic_scenarios))
    
    # Set target emotion
    target_emotion = args.emotion 
    
    num_samples = args.num_samples

    # Load names from file
    names = json.load(open(args.names))

    generated_scenarios = []

    # Generate full scenario for each basic scenario
    for i, basic_scenario in enumerate(basic_scenarios):
        
        print(f"\nBasic Situation:\n{basic_scenario["scenario"]}\n")
        response = generate_scenario(
            basic_scenario=basic_scenario["scenario"],
            target_emotion=target_emotion,
            names=names,
            num_samples=num_samples
        )

        scenario_dict = basic_scenario
        scenario_dict["target_emotion"] = target_emotion
        scenario_dict["response"] = response

        generated_scenarios.append(scenario_dict)

    # Save responses
    output = json.dumps(generated_scenarios, indent=4)
    with open("responses.json", "w") as outfile:
        outfile.write(output)
