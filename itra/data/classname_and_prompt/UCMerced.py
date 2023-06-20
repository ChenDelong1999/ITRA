new_classes = {
    "mediumresidential": "medium residential",
    "harbor": "harbor",
    "parkinglot": "parking lot",
    "golfcourse": "golf course",
    "storagetanks": "storage tanks",
    "denseresidential": "dense residential",
    "agricultural": "agricultural",
    "runway": "runway",
    "intersection": "intersection",
    "tenniscourt": "tennis court",
    "airplane": "airplane",
    "buildings": "buildings",
    "baseballdiamond": "baseball diamond",
    "beach": "beach",
    "river": "river",
    "overpass": "overpass",
    "forest": "forest",
    "mobilehomepark": "mobile home park",
    "chaparral": "chaparral",
    "sparseresidential": "sparse residential",
    "freeway": "freeway",
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