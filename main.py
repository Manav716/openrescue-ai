from env.disaster_env import DisasterEnv
from env.render import Renderer
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from transformers import pipeline
import torch
import time
import os
import re

print("=== INITIALIZING OPENRESCUE AI ===")

# 1. Load Hugging Face NLP Brain
print("Loading NLP Transformer...")
device_id = 0 if torch.cuda.is_available() else -1
nlp_brain = pipeline(
    "zero-shot-classification", 
    model="facebook/bart-large-mnli",
    device=device_id
)
candidate_labels = ["Critical Rescue Target", "Lethal Hazard", "Routine Exploration"]
print("NLP Brain Ready!")

# 2. Setup Environment and Renderer
base_env = DisasterEnv()
wrapped_env = FlattenObservation(base_env) # Flatten for PPO compatibility
renderer = Renderer()

# 3. Load the Trained PPO Brain
model_path = "models/best_model"
if not os.path.exists(model_path + ".zip"):
    print(f"\n[ERROR] Could not find the trained model at '{model_path}.zip'")
    print("Please download it from Colab and place it in the 'models/' folder.")
    renderer.close()
    exit()

print("Loading PPO RL Brain...")
model = PPO.load(model_path, env=wrapped_env)
print("RL Brain Ready!\n")

# 4. Run Simulation
obs, info = wrapped_env.reset()
done = False
truncated = False
step_counter = 0
printed_messages_count = 0

# --- THE MAGIC: NLP Override State ---
override_target = None

print("=== STARTING MISSION ===")

while not done and not truncated:
    
    # --- NEURO-SYMBOLIC OVERRIDE LOGIC ---
    if override_target is not None:
        target_x, target_y = override_target
        current_x, current_y = base_env.rescue_agent.pos
        
        # Symbolic Pathfinding to force the agent to the NLP target
        if current_x > target_x: action = 0
        elif current_x < target_x: action = 1
        elif current_y > target_y: action = 2
        elif current_y < target_y: action = 3
        else:
            # Target Reached! Return control to PPO Agent.
            override_target = None
            action, _states = model.predict(obs, deterministic=True)
    else:
        # Standard RL Autonomous Movement
        action, _states = model.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, done, truncated, info = wrapped_env.step(action)
    step_counter += 1

    # Render Visuals
    renderer.draw(base_env.grid)

    # Process Drone Communications via NLP
    current_messages = base_env.messages
    if len(current_messages) > printed_messages_count:
        new_messages = current_messages[printed_messages_count:]
        
        for msg in new_messages:
            # AI Classification
            analysis = nlp_brain(msg, candidate_labels)
            top_category = analysis['labels'][0]
            confidence = analysis['scores'][0] * 100
            
            print(f"\n[Step {step_counter}] 🚁 DRONE ALERT: {msg}")
            
            if top_category == "Lethal Hazard":
                print(f"   ⚠️ NLP CLASSIFICATION: {top_category} ({confidence:.1f}%)")
                print(f"   🤖 COMMAND: Logging hazard for future avoidance.")
                
            elif top_category == "Critical Rescue Target":
                print(f"   🚨 NLP CLASSIFICATION: {top_category} ({confidence:.1f}%)")
                
                # Extract coordinates using Regular Expressions!
                match = re.search(r'\((\d+),\s*(\d+)\)', msg)
                if match:
                    override_target = (int(match.group(1)), int(match.group(2)))
                    print(f"   ⚡ COMMAND OVERRIDE: Taking control of Rescue Agent!")
                    print(f"   ⚡ ROUTING TO: {override_target}")
                    
            else:
                print(f"   ℹ️ NLP CLASSIFICATION: {top_category} ({confidence:.1f}%)")
        
        printed_messages_count = len(current_messages)

    # Slow down so humans can watch the Pygame window!
    time.sleep(0.3)

print("\n=== MISSION COMPLETE ===")
renderer.close()