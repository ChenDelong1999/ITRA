new_classes = {
    'Forest':'forest',
    'PermanentCrop':'permanent crop land',
    'Residential':'residential buildings or homes or apartments',
    'River':'river',
    'Pasture':'pasture land',
    'SeaLake':'lake or sea',
    'HerbaceousVegetation':'brushland or shrubland',
    'AnnualCrop':'annual crop land',
    'Industrial':'industrial buildings or commercial buildings',
    'Highway':'highway or road'

}
# classes 对应文本提示中的类名
# file_class_name 对应数据文件的类名
classes = list(new_classes.values())
file_class_names = list(new_classes.keys())

templates = [
    'a centered satellite photo of {}.',
    'a centered satellite photo of a {}.',
    'a centered satellite photo of the {}.',
]