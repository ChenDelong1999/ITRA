new_classes = {
    "Airport": "airport",
    "Beach": "beach",
    "Bridge": "bridge",
    "Commercial": "Commercial",
    "Desert": "desert",
    "Farmland": "farmland",
    "footballField": "footballField",
    "Forest": "forest",
    "Industrial": "industrial",
    "Meadow": "meadow",
    "Mountain": "mountain",
    "Park": "park",
    "Parking": "parking",
    "Pond": "pond",
    "Port": "port",
    "railwayStation": "railwayStation",
    "Residential": "residential",
    "River": "river",
    "Viaduct": "viaduct",
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