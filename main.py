from env.disaster_env import DisasterEnv
from env.render import Renderer
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from transformers import pipeline
import torch
import time
import os

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

print("=== STARTING MISSION ===")

while not done and not truncated:
    # Get action from AI
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
                print(f"   🤖 COMMAND: Rerouting Rescue Agent to avoid coordinates.")
            elif top_category == "Critical Rescue Target":
                print(f"   🚨 NLP CLASSIFICATION: {top_category} ({confidence:.1f}%)")
                print(f"   🤖 COMMAND: High priority! Pinging coordinates to Rescue Agent.")
            else:
                print(f"   ℹ️ NLP CLASSIFICATION: {top_category} ({confidence:.1f}%)")
        
        printed_messages_count = len(current_messages)

    # Slow down so humans can watch the Pygame window!
    time.sleep(0.3)

print("\n=== MISSION COMPLETE ===")
renderer.close()