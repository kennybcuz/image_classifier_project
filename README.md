# Udacity Image Classifier Project
run both train.py and predict.py from the command line.

train.py example:
python train.py --data_dir flowers --save_dir sample_checkpoint.pth --gpu --hidden_units 4096 2048 512 --epochs 4 --learning_rate 0.001 --arch vgg

predict.py example:
python predict.py --checkpoint sample_checkpoint.pth --input flowers/test/102/image_08004.jpg --top_k 3 --category_names cat_to_name.json
