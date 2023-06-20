new_classes = {
    "denseforest": "dense forest",
    "highbuildings": "high buildings",
    "railway": "railway",
    "sparseforest": "sparse forest",
    "grassland": "grassland",
    "lowbuildings": "low buildings",
    "residentialarea": "residential area",
    "stroagetanks": "storage tanks",
    "harbor": "harbor",
    "overpass": "overpass",
    "roads": "roads",
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