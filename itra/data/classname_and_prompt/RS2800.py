new_classes = {
    "RiverLake": "river or lake",
    "Parking": "parking",
    "Industry": "industry",
    "Forest": "forest",
    "Field": "field",
    "Resident": "resident",
    "Grass": "grass",
}
# new_classes = {
#     "dRiverLake": "river or lake",
#     "gParking": "parking",
#     "cIndustry": "industry",
#     "eForest": "forest",
#     "bField": "field",
#     "fResident": "resident",
#     "aGrass": "grass",
# }
classes = list(new_classes.values())
file_class_names = list(new_classes.keys())

templates = [
    'a centered satellite photo of a {}.',
    'a centered satellite photo of the {} taken from above.',
    'an overhead view of the {}.',
    'an overhead view of a {}.',
    'an overhead view of {}.',
]