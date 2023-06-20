new_classes = {
    "meadow": "meadow",
    "water": "water",
    "harbor": "harbor",
    "pond": "pond",
    "park": "park",
    "commercial": "commercial",
    "idle_land": "idle land",
    "river": "river",
    "residential": "residential",
    "overpass": "overpass",
    "agriculture": "agriculture",
    "industrial": "industrial",
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