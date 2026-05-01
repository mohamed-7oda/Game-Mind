import os
import json
import numpy as np
from config.classes import INVERSE_EVENT_DICTIONARY_V2


def predictions2json(predictions_half_1, output_path, framerate=2, half=1):
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, "Predictions-v2.json")
    
    frames_half_1, class_half_1 = np.where(predictions_half_1 >= 0)
    
    json_data = dict()
    json_data["predictions"] = list()
    
    for frame_index, class_index in zip(frames_half_1, class_half_1):
        confidence = predictions_half_1[frame_index, class_index]
        
        seconds = int((frame_index // framerate) % 60)
        minutes = int((frame_index // framerate) // 60)
        
        prediction_data = dict()
        prediction_data["gameTime"] = f"{half} - {minutes}:{seconds}"
        prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[class_index]
        prediction_data["position"] = str(int((frame_index / framerate) * 1000))
        prediction_data["half"] = str(half)
        prediction_data["confidence"] = str(confidence)
        
        json_data["predictions"].append(prediction_data)
    
    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file, indent=4)
    
    print(f"Predictions saved to {output_file_path}")