#!/bin/sh
#SBATCH --nodes=1
##SBATCH --nodelist=above203
#SBATCH --ntasks=1
#SBATCH --qos=long
#SBATCH --time=10-0
##SBATCH --time=1-0

##SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=3G
#SBATCH -G4
##SBATCH -G0

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Rscript /home/spotter5/cnn_mapping/v4/burned_area_calculations_landsat.r $1 
#Rscript /home/spotter5/cnn_mapping/v4/burned_area_calculations_landsat_om_com.r $1 
# Rscript /home/spotter5/cnn_mapping/v4/fire_cci_raw_extract.r $1 
# python /home/spotter5/cnn_mapping/v5/train_cnn_dNBR.py 
# python /home/spotter5/cnn_mapping/v5/train_cnn_VI.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/train_cnn_VI_unet1.py 
# python /home/spotter5/cnn_mapping/eurasia_cnn_fire_mapping/train_cnn_VI_net3.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/ndsi_download.py 

# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_training_landsat_no_shift.py 
#python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_training_landsat_no_shift_mtbs.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_training_landsat_no_shift_anna.py 

# python /home/spotter5/cnn_mapping/eurasia_cnn_fire_mapping/download_anna_ndsi_final.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_nbac_ndsi_final.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_mtbs_ndsi_final.py 

# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_nbac_ndsi_final_sliding.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_mtbs_ndsi_final_sliding.py 
# python /home/spotter5/cnn_mapping/eurasia_cnn_fire_mapping/download_anna_ndsi_final_sliding.py 

# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_training_landsat_85.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/swin_train_cnn_VI.py 

# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/tif_to_parquet.py 



# python /home/spotter5/cnn_mapping/eurasia_cnn_fire_mapping/composite_with_snow_cover_try_faster_keep_date.py
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/mtbs_composite_with_snow_cover_try_faster_keep_date.py
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/nbac_composite_with_snow_cover_try_faster_keep_date.py

python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/train_cnn_VI.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/train_cnn_VI_ndsi.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/train_cnn_VI_ndsi_sliding.py 

# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/na_xgboost.py 

# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/train_cnn_VI_unet1.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/train_cnn_VI_unet3.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/train_cnn_VI_unet1_dnbr.py 

# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/train_cnn_VI_unet3_dnbr.py 

# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/swin.py 
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/transunet.py 


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
# python /home/spotter5/cnn_mapping/na_cnn_fire_mapping/download_training_landsat_no_shift.py












