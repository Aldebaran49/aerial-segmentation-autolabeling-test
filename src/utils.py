import numpy as np

# Cityscapes:
# 0: Road, 1: Sidewalk, 2: Building, 3: Wall, 4: Fence, 5: Pole,
# 6: Traffic Light, 7: Traffic Sign, 8: Vegetation, 9: Terrain, 
# 10: Sky, 11: Person, 12: Rider, 13: Car, 14: Truck, 15: Bus, 
# 16: Train, 17: Motorcycle, 18: Bicycle
#
# UAVid Classes:
# 0: Clutter, 1: Building, 2: Road, 3: Static_Car, 
# 4: Tree, 5: Low_Veg, 6: Human, 7: Moving_Car
#
# Aeroscapes Classes:
# 0: Background, 1: Person, 2: Bike, 3: Car, 4: Drone, 5: Boat, 
# 6: Animal, 7: Obstacle, 8: Construction, 9: Vegetation, 10: Road, 11: Sky

CITYSCAPES_TO_UAVID = {
    0: 2,   # Road - Road
    1: 2,   # Sidewalk - Road
    2: 1,   # Building - Building
    3: 0,   # Wall - Clutter
    4: 0,   # Fence - Clutter
    5: 0,   # Pole - Clutter
    6: 0,   # Traffic Light - Clutter
    7: 0,   # Traffic Sign - Clutter
    8: 4,   # Vegetation - Tree
    9: 5,   # Terrain - Low Vegetation
    10: 0,  # Sky - Clutter
    11: 6,  # Person - Human
    12: 6,  # Rider - Human
    13: 7,  # Car - Moving Car
    14: 7,  # Truck - Moving Car
    15: 7,  # Bus - Moving Car
    16: 0,  # Train - Clutter
    17: 0,  # Motorcycle - Clutter
    18: 0   # Bicycle - Clutter
}

CITYSCAPES_TO_AEROSCAPES = {
    0: 10,  # Road - Road
    1: 10,  # Sidewalk - Road
    2: 8,   # Building - Construction
    3: 8,   # Wall - Construction
    4: 8,   # Fence - Construction
    5: 7,   # Pole - Obstacle
    6: 7,   # Traffic Light - Obstacle
    7: 7,   # Traffic Sign - Obstacle
    8: 9,   # Vegetation - Vegetation
    9: 9,   # Terrain - Vegetation
    10: 11, # Sky - Sky
    11: 1,  # Person - Person
    12: 1,  # Rider - Person
    13: 3,  # Car - Car
    14: 3,  # Truck - Car
    15: 3,  # Bus - Car
    16: 0,  # Train - Background
    17: 2,  # Motorcycle - Bike
    18: 2   # Bicycle - Bike
}

def map_mask(pred_mask, mapping_dict):
    """
    Нужен для конвертации маски в формат нужного датасета в зависимости от mapping_dict.
    Args:
        pred_mask (np.ndarray): Маска (H, W) Cityscapes.
        mapping_dict (dict): Словарь {old_id: new_id}.
        
    Returns:
        np.ndarray: Нужная маска (H, W).
    """
    lookup_table = np.zeros(20, dtype=np.uint8)
    
    for old_id, new_id in mapping_dict.items():
        lookup_table[old_id] = new_id

    mapped_mask = lookup_table[pred_mask]
    
    return mapped_mask
 

# Grounded-SAM promts
# ID: "text prompt"
#
# UAVID PROMPTS
UAVID_SAM_CONFIG = {
    "prompts": {
        # 0: "clutter",
        1: "building, skyscraper, roof, concrete structure, architecture, house, apartment",
        2: "road, street, asphalt, highway, pavement, lane, driveway",
        #3: "", 
        4: "tree, tree crown, green tree",
        5: "grass, lawn, vegetation, bush, field", 
        6: "person, pedestrian, human",
        7: "car, moving car, driving car, automobile, truck, bus"
    },
    # Порядок: Трава/Дорога -> Здания -> Деревья -> Мелочь
    "z_order": [5, 2, 1, 4, 7, 6] 
}

# AEROSCAPES PROMPTS
AEROSCAPES_SAM_CONFIG = {
    "prompts": {
        #0: "background",
        1: "person, pedestrian, human walking",
        2: "bicycle, bike, cyclist",
        3: "car, automobile, van, truck", 
        4: "quadcopter drone, small uav flying", 
        5: "boat on water, ship on water",
        6: "animal, dog, cow, sheep",
        7: "obstacle, cone, barrier, pile",
        
        8: "building, railway bridge, overpass, concrete bridge, skyscraper, glass building, bridge, overpass, concrete structure, house, concrete pillar",
        
        9: "vegetation, tree, grass, bush, plants, field, small bush",
        
        10: "road, asphalt, street, highway, pavement",
        11: "sky, clouds, overcast sky"
    },
    "z_order": [11, 4, 5, 6, 0, 9, 7, 10, 8, 3, 2, 1]
}