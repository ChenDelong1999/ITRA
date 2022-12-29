# VISSL disk_folder
cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_euro_sat_data_files.py -i /data/Datasets/euro_sat/ -o /data/Datasets/euro_sat -d

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_food101_data_files.py -i /data/Datasets/food101/ -o /data/Datasets/food101 -d

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_kitti_dist_data_files.py -i /data/Datasets/kitti_dist/ -o /data/Datasets/kitti_dist -d

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_patch_camelyon_data_files.py -i /data/Datasets/patch_camelyon/ -o /data/Datasets/patch_camelyon -d

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_food101_data_files.py -i /data/Datasets/food101/ -o /data/Datasets/food101 -d

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_food101_data_files.py -i /data/Datasets/food101/ -o /data/Datasets/food101 -d

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_ucf101_data_files.py -i /data/Datasets/ucf101-vissl/ -o /data/Datasets/ucf101-vissl -d

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_stanford_cars_data_files.py -i /data/Datasets/stanford_cars-vissl/ -o /data/Datasets/stanford_cars-vissl -d

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_k700_data_files.py -i /data/Datasets/k700/ -o /data/Datasets/k700 -d

# VISSL disk_filelist

cd '/data/codes/vissl'
export PYTHONPATH="."
python extra_scripts/datasets/create_sun397_data_files.py -i /data/Datasets/sun397/ -o /data/Datasets/sun397 -d




# torchvision

# python
import torchvision
#dataset = torchvision.datasets.Country211(root='/data/Datasets/country211-torchvision', download='True')
# dataset = torchvision.datasets.GTSRB(root='/data/Datasets/GTSRB-torchvision', download='True')
# dataset = torchvision.datasets.OxfordIIITPet(root='/data/Datasets/OxfordIIITPet', download='True')
# dataset = torchvision.datasets.RenderedSST2(root='/data/Datasets/RenderedSST2', download='True')
#dataset = torchvision.datasets.SVHN(root='/data/Datasets/SVHN', download='True')
dataset = torchvision.datasets.MNIST(root='/data/Datasets/MNIST', download='True')

cd /data/Datasets
oss cp 's3://research-model-hh-b/Dataset/fine_grained/SUN397.tar.gz' 'SUN397.tar.gz'
tar -zxvf 'SUN397.tar.gz'

cd /data/Datasets
oss cp 's3://research-model-hh-b/Dataset/fine_grained/dtd-r1.0.1.tar.gz' 'DTD.tar.gz'
tar -zxvf 'DTD.tar.gz'

cd /data/Datasets
oss cp 's3://research-model-hh-b/Dataset/fine_grained/food-101.tar.gz' 'food-101.tar.gz'
tar -zxvf 'food-101.tar.gz'

cd /data/Datasets
oss cp 's3://research-model-hh-b/Dataset/fine_grained/102flowers.tgz' 'flowers102.tgz'
tar -zxvf 'flowers102-oss.tgz'

cd /data/Datasets
mkdir 'fgvc-aircraft'
cd 'fgvc-aircraft'
oss cp 's3://research-model-hh-b/Dataset/fine_grained/aircraft/fgvc-aircraft-2013b-annotations.tar.gz' 'fgvc-aircraft-2013b-annotations.tar.gz'
oss cp 's3://research-model-hh-b/Dataset/fine_grained/aircraft/fgvc-aircraft-2013b.tar.gz' 'fgvc-aircraft-2013b.tar.gz'
tar -zxvf 'fgvc-aircraft-2013b-annotations.tar.gz'
tar -zxvf 'fgvc-aircraft-2013b.tar.gz'
cd ..


cd /data/Datasets
mkdir 'stanford_cars'
cd 'stanford_cars'
oss cp 's3://research-model-hh-b/Dataset/fine_grained/stanford_cars/car_devkit.tgz' 'car_devkit.tgz'
oss cp 's3://research-model-hh-b/Dataset/fine_grained/stanford_cars/car_ims.tgz' 'car_ims.tgz'
tar -zxvf 'car_devkit.tgz'
tar -zxvf 'car_ims.tgz'
cd ..

cd /data/Datasets
mkdir 'caltech101'
cd 'caltech101'
oss cp 's3://research-model-hh-b/Dataset/fine_grained/caltech101/101_ObjectCategories.tar.gz' '101_ObjectCategories.tar.gz'
oss cp 's3://research-model-hh-b/Dataset/fine_grained/caltech101/Annotations.tar' 'Annotations.tar'
tar -zxvf '101_ObjectCategories.tar.gz'
tar -xvf 'Annotations.tar'
cd ..

