#!/bin/sh
#SBATCH --nodes=1
##SBATCH --nodelist=gpu002
#SBATCH --ntasks=1
#SBATCH --qos=long
#SBATCH --time=5-0
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
##SBATCH -G4
#SBATCH -G0



# Rscript /home/spotter5/cnn_mapping/v4/burned_area_calculations_landsat.r $1 
#Rscript /home/spotter5/cnn_mapping/v4/burned_area_calculations_landsat_om_com.r $1 
# Rscript /home/spotter5/cnn_mapping/v4/fire_cci_raw_extract.r $1 
# python /home/spotter5/cnn_mapping/v5/train_cnn_dNBR.py 
# python /home/spotter5/cnn_mapping/v5/train_cnn_VI.py 
# python /home/spotter5/cnn_mapping/v5/train_cnn_VI_unburned.py 
# python /home/spotter5/cnn_mapping/v5/train_cnn_VI_2019.py 
# python /home/spotter5/cnn_mapping/v5/transunet_train_cnn_VI_2019.py 
# python /home/spotter5/cnn_mapping/v5/transunet_train_cnn_VI.py 
# python /home/spotter5/cnn_mapping/v5/swin_train_cnn_VI.py 

# python /home/spotter5/cnn_mapping/v5/swin_train_cnn_VI_2019.py 
# python /home/spotter5/cnn_mapping/v5/download_training_landsat_85.py 
# python /home/spotter5/cnn_mapping/v5/split_to_chunks_nbac.py 
# python /home/spotter5/cnn_mapping/v5/train_cnn_VI_regularize.py 
# python /home/spotter5/cnn_mapping/v5/train_cnn_VI_regularize_85.py 

# python /home/spotter5/cnn_mapping/v5/train_cnn_VI_3p_regularize.py
# python /home/spotter5/cnn_mapping/v5/tif_to_parquet.py 
python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_training_landsat_no_shift.py












