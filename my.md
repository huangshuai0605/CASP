#Casp cub200
python -m pdb train_my.py  -dataset cub200 -base_mode ft_dot -new_mode avg_cos -gamma 0.1 -lr_base 1e-2 -lr_new 1e-3 -decay 0.0005 -epochs_base 30 -epochs_new 0 -schedule Cosine  -gpu 0 -temperature 16 -start_session 0 -batch_size_base 64 -seed 1 -vit  -out 'PriViLege' -dataroot /workspace/huangshuai/CEC-CVPR2021/data/ -mix_layer 0 -mix 0.5 -project Casp

#Casp cifar100
python -m pdb train_my.py  -dataset cifar100 -base_mode ft_dot -new_mode avg_cos -gamma 0.1 -lr_base 1e-2 -lr_new 1e-3 -decay 0.0005 -epochs_base 30 -epochs_new 0 -schedule Cosine  -gpu 0 -temperature 16 -start_session 0 -batch_size_base 64 -seed 1 -vit  -out 'PriViLege' -dataroot /workspace/huangshuai/CEC-CVPR2021/data/ -mix_layer 5 -mix 0.05 -project Casp
