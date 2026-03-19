
from dialz import Dataset, SteeringVector
from src.core.config import X_AXIS_ID, Y_AXIS_ID

def train_cultural_vector(model, train_data, axis='X', system_prompt=""):
    """
    Trains a steering vector based on X or Y axis items in train_data.
    """
    dataset = Dataset()
    target_ids = X_AXIS_ID if axis == 'X' else Y_AXIS_ID
    
    for sample in train_data:
        if sample['wvs_id'] not in target_ids:
            continue
            
        situation = sample.get('scenario_text')
        low_pole = sample['options'][sample['mapping']['low_pole']]
        high_pole = sample['options'][sample['mapping']['high_pole']]
        
        # Dialz expects pairs: (negative, positive)
        dataset.add_entry(f"{system_prompt} \n{situation} Answer: {low_pole}", f"{system_prompt} \n{situation} Answer: {high_pole}")
        
    steering_vector = SteeringVector.train(model, dataset, method="mean_diff")
    return steering_vector
