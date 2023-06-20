
new_classes = {
    "container": "container",
    "sea": "sea",
    "highway": "highway",
    "avenue": "avenue",
    "bare_land": "bare land",
    "sparse_forest": "sparse forest",
    "residents": "residents",
    "parkinglot": "parkinglot",
    "sandbeach": "sandbeach",
    "storage_room": "storage room",
    "mangrove": "mangrove",
    "sapling": "sapling",
    "airplane": "airplane",
    "artificial_grassland": "artificial grassland",
    "town": "town",
    "snow_mountain": "snow mountain",
    "river": "river",
    "stream": "stream",
    "city_building": "city building",
    "green_farmland": "green farmland",
    "bridge": "bridge",
    "dry_farm": "dry farm",
    "forest": "forest",
    "pipeline": "pipeline",
    "shrubwood": "shrubwood",
    "crossroads": "crossroads",
    "dam": "dam",
    "coastline": "coastline",
    "lakeshore": "lakeshore",
    "hirst": "hirst",
    "mountain": "mountain",
    "airport_runway": "airport runway",
    "marina": "marina",
    "river_protection_forest": "river protection forest",
    "desert": "desert",
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