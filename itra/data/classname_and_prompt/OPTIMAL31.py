
new_classes = {
    "airplane": "airplane",
    "airport": "airport",
    "baseball_diamond": "baseball diamond",
    "basketball_court": "basketball court",
    "beach": "beach",
    "bridge": "bridge",
    "chaparral": "chaparral",
    "church": "church",
    "circular_farmland": "circular farmland",
    "commercial_area": "commercial area",
    "dense_residential": "dense residential",
    "desert": "desert",
    "forest": "forest",
    "freeway": "freeway",
    "golf_course": "golf course",
    "ground_track_field": "ground track field",
    "harbor": "harbor",
    "industrial_area": "industrial area",
    "intersection": "intersection",
    "island": "island",
    "lake": "lake",
    "meadow": "meadow",
    "medium_residential": "medium residential",
    "mobile_home_park": "mobile home park",
    "mountain": "mountain",
    "overpass": "overpass",
    "parking_lot": "parking lot",
    "railway": "railway",
    "rectangular_farmland": "rectangular farmland",
    "roundabout": "roundabout",
    "runway": "runway",
}
classes = list(new_classes.values())
file_class_names = list(new_classes.keys())

templates = [
    'a centered satellite photo of a {}.',
    'a centered satellite photo of the {} taken from above.',
    'an overhead view of the {}.',
    'an overhead view of a {}.',
    'an overhead view of {}.',
]