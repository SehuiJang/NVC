#### OfficeHome

# SDAT
# pretrained models
# |_ officehome_ar2cl_sdat.pth: https://drive.google.com/file/d/1AB-DqoWRZiTOrm087Y7x_loU0Mbcvof1/view?usp=drive_link
# |_ officehome_ar2pr_sdat.pth: https://drive.google.com/file/d/1sofFz280gvCPKQp4VHtT17i14LRt-k2v/view?usp=drive_link
# |_ officehome_ar2rw_sdat.pth: https://drive.google.com/file/d/1r-fdYjnTXdry-9nWCLxbJnEKrPI34o7p/view?usp=drive_link
# |_ officehome_cl2ar_sdat.pth: https://drive.google.com/file/d/1yfFbD6IzZGJyzc5cSUGadOQEurSkRoJ1/view?usp=drive_link
# |_ officehome_cl2pr_sdat.pth: https://drive.google.com/file/d/1gHVv_415MTluK7NNftpPeZ0bVVE0iRx_/view?usp=drive_link
# |_ officehome_cl2rw_sdat.pth: https://drive.google.com/file/d/1_HC7eySl1W8szjz2JatBxveXwFde8ess/view?usp=drive_link
# |_ officehome_pr2ar_sdat.pth: https://drive.google.com/file/d/1zZg7UCO_iXhjsfqnTE2F9HaNNmDtGgXL/view?usp=drive_link
# |_ officehome_pr2cl_sdat.pth: https://drive.google.com/file/d/1yl7353WIqTJVO9b9Si-s7BkrNOSGwZDv/view?usp=drive_link
# |_ officehome_pr2rw_sdat.pth: https://drive.google.com/file/d/15r1wVJNk0tMSf1fq0I_tvwrJ-QWNhOWj/view?usp=drive_link
# |_ officehome_rw2ar_sdat.pth: https://drive.google.com/file/d/1uvg8BY313tjQxP0dLqihDwa5AEgjrUYg/view?usp=drive_link
# |_ officehome_rw2cl_sdat.pth: https://drive.google.com/file/d/1iMo8zz7b3NGun4UOv6_VUyJ_paxXV8kH/view?usp=drive_link
# |_ officehome_rw2pr_sdat.pth: https://drive.google.com/file/d/1Lytw8EupNsL9Pr88egnEI0nmwvudel6z/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Cl --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Ar2Cl/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Pr --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Ar2Pr/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Rw --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Ar2Rw/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Ar --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Cl2Ar/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Pr --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Cl2Pr/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Rw --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Cl2Rw/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Ar --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Pr2Ar/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Cl --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Pr2Cl/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Rw --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Pr2Rw/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Ar --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Rw2Ar/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Cl --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Rw2Cl/sdat ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Pr --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Rw2Pr/sdat ;

# SDAT+NVC
# pretrained models
# |_ officehome_ar2cl_nvc.pth: https://drive.google.com/file/d/13DuKIeVKxUwyKI-Xo1JoOnH7LRVzdWfe/view?usp=drive_link
# |_ officehome_ar2pr_nvc.pth: https://drive.google.com/file/d/1RsgcRArWwTC4RRMOhgczbFaGNL2ivkpp/view?usp=drive_link
# |_ officehome_ar2rw_nvc.pth: https://drive.google.com/file/d/11t9gFhKieA9VfEONazrTAQgpNeMeiuSt/view?usp=drive_link
# |_ officehome_cl2ar_nvc.pth: https://drive.google.com/file/d/174-BNnfGSMNYXPpK42f6vDZkHjOuNMmz/view?usp=drive_link
# |_ officehome_cl2pr_nvc.pth: https://drive.google.com/file/d/1NqU3LdAXmCqZoyOOX0KpqRW3MD-1kvXL/view?usp=drive_link
# |_ officehome_cl2rw_nvc.pth: https://drive.google.com/file/d/1c7GSyG9uhay6vm4PmXaS-VDzqSH2JANI/view?usp=drive_link
# |_ officehome_pr2ar_nvc.pth: https://drive.google.com/file/d/17V7cTh5BzEmizbhcWSvnjUxAjcm6VB2I/view?usp=drive_link
# |_ officehome_pr2cl_nvc.pth: https://drive.google.com/file/d/17tZUmFVJstRLo6rHFNwUP56OJSbCifb1/view?usp=drive_link
# |_ officehome_pr2rw_nvc.pth: https://drive.google.com/file/d/1giK4hyQ4xzQ121jP4bYwHGH6EK_or7mF/view?usp=drive_link
# |_ officehome_rw2ar_nvc.pth: https://drive.google.com/file/d/1sYSgQWEcJqsAboAxkdIjnOXbzawF0LqD/view?usp=drive_link
# |_ officehome_rw2cl_nvc.pth: https://drive.google.com/file/d/11stQ4dNekO_zlnDtXShp2l1ecSWyoYU9/view?usp=drive_link
# |_ officehome_rw2pr_nvc.pth: https://drive.google.com/file/d/1RInb2q00z_JZVL2IuFzYjw7Tm2s60_zx/view?usp=drive_link
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Cl --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Ar2Cl/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Pr --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Ar2Pr/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Ar -t Rw --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Ar2Rw/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Ar --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Cl2Ar/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Pr --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Cl2Pr/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Cl -t Rw --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Cl2Rw/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Ar --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Pr2Ar/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Cl --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Pr2Cl/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Pr -t Rw --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Pr2Rw/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Ar --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Rw2Ar/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Cl --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Rw2Cl/nvc ;
CUDA_VISIBLE_DEVICES=0,1,2 python cdan_mcc_sdat_nvc.py /your/path/to/Office-Home -d OfficeHome -s Rw -t Pr --epochs 35 -b 96 -i 50 -p 25 --no-pool --lr 0.010 --seed 0 -a vit_base_patch16_224 --rho 0.02     --denorm_and_toPIL --gaussian_blur       --triplet_type latentv2 --triplet_coef 0.5 --triplet_temp 0.5     --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32         --log /your/target_directory/to/save/log_and_checkpoints/Rw2Pr/nvc ;
