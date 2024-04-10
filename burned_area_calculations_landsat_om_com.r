library(terra)
library(sf)
library(tidyverse)

args = commandArgs(TRUE)



in_year = args[1]

out_path = '/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/area_comparisons/landsat_VI2'
dir.create(out_path, recursive = T)

#path to ak
ak = read_sf('/explore/nobackup/people/spotter5/auxillary/Features/Alaska/Boundaries/alaska.shp')

#path to canada
ca = read_sf('/explore/nobackup/people/spotter5/auxillary/Features/Canada/Provinces/Canada_Provinces.shp') 

#path to landsat ba
# landsat_ba_path = '/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/final_predictions_VI'
landsat_ba_path = '/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/final_predictions_VI2'

in_files = list.files(landsat_ba_path, full.names = F, pattern = ".tif$")

#files to loop through
# years = c(2004, 2005, 2014, 2015)

# years = c(2014, 2015)
#for combination of all years
combined = list()

for(f in in_files){
    
     new_f = str_replace(f, '.tif', '.csv')
    
    # if(f == 'predicted_2004_constrained0000053760-0000080640.tif'){
    
     
    if (file.exists( file.path(out_path, new_f)) == FALSE){
        #get year
        year = str_split(f, '_')[[1]][2]
        
        if(year == as.character(in_year)){
                #us
            fire_cci = rast(file.path(landsat_ba_path, f))[[1]]
            fire_cci[fire_cci == 0] = NA
            fire_cci[fire_cci > 0] = 1

            #gabam
            mcd64a1 = rast(file.path(landsat_ba_path, f))[[2]]
            mcd64a1[mcd64a1 == 0] = NA
            mcd64a1[mcd64a1 > 0] = 1

            #lfdb
            lfdb = rast(file.path(landsat_ba_path, f))[[3]]
            lfdb[lfdb == 0] = NA
            lfdb[lfdb > 0] = 1

            #now get intersection area of us and gabam with lfdb
            fire_cci_int = rast(file.path(landsat_ba_path, f))[[1]]
            mcd64a1_int = rast(file.path(landsat_ba_path, f))[[2]]
            # fire_cci_int[fire_cci_int == 0] = NA
            mcd64a1_int[ mcd64a1_int == 0] = NA
            mcd64a1_int[ mcd64a1_int == 1] = 9




            # fire_cci_int[fire_cci_int > 500] = 1 #500 is 0.5 probability is fire, change to 1 if greater and burned
            fire_cci_int[(lfdb == 1) & (fire_cci_int > 500)] = 1
            fire_cci_int[fire_cci_int !=1] = NA

            # mcd64a1_int[mcd64a1_int == 1] = 9 #gabam is already a 1 if it burned 
            mcd64a1_int[(lfdb == 1) & (mcd64a1_int == 9)] = 1 #then say if lfdb is also one make it a 9
            mcd64a1_int[mcd64a1_int !=1] = NA #then say if it is not a 9 make it a NA, so just retain intersections


            #now get where us and gabam have fire but lfdb does not
            fire_cci_have = rast(file.path(landsat_ba_path, f))[[1]]
            mcd64a1_have = rast(file.path(landsat_ba_path, f))[[2]]
    #         fire_cci_int[fire_cci_int == 0] = NA
            mcd64a1_have[ mcd64a1_have == 0] = NA

            # fire_cci_have[fire_cci_have == 0] = NA
            fire_cci_have[(is.na(lfdb)) & (fire_cci_have > 500)] = 1 #we have fire when greater than 500
            fire_cci_have[fire_cci_have !=1] = NA

            # mcd64a1_have[mcd64a1_have == 0] = NA
            mcd64a1_have[mcd64a1_have == 1] = 9 #gabam is already a 1 I think 
            mcd64a1_have[(is.na(lfdb)) & (mcd64a1_have == 9)] = 1 #get where no data for lfdb but gabam has fire
            mcd64a1_have[mcd64a1_have !=1] = NA

            #now get where us and gabam do not have fire but lfdb does
            fire_cci_not = rast(file.path(landsat_ba_path, f))[[1]]
            mcd64a1_not = rast(file.path(landsat_ba_path, f))[[2]]
            # mcd64a1_not[mcd64a1_not == 0] = NA

            # fire_cci_not[fire_cci_not == 0] = NA
            # fire_cci_not[fire_cci_not < 500] = NA #less than 500 means it didnt burn for us (< 0.5 probability)
            # fire_cci_not[fire_cci_not < 500] = 0 # we don't have fire
            # fire_cci_not[fire_cci_not > 500] = 1# we do have fire

            # fire_cci_not[fire_cci_not < 500] = 0
            # fire_cci_not[fire_cci_not > 500] = 1


            fire_cci_not[(lfdb == 1) & (fire_cci_not == 0)] = 1 #if lfdb is 1 it has fire, if less then 0.5 we do not
            fire_cci_not[fire_cci_not !=1] = NA

            # mcd64a1_not[mcd64a1_not == 0] = NA
            mcd64a1_not[mcd64a1_not == 1] = 9 #gabam is already a 1, so change burned area to 9
            mcd64a1_not[(lfdb == 1) & (mcd64a1_not ==0)] = 1 # if lfdb is 1 and gabam is not = 9 it means lfdb burned but gabam did not, change to 1
            mcd64a1_not[mcd64a1_not !=1] = NA # if it is not equal to 1 (eg 9 which burned), convert to NA

            #get cellsize
            res = xres(fire_cci)

            #convert to areas in square meters
            fire_cci = fire_cci * res^2
            mcd64a1 = mcd64a1 * res^2
            lfdb = lfdb * res^2

            fire_cci_int = fire_cci_int * res^2
            fire_cci_have = fire_cci_have * res^2
            fire_cci_not = fire_cci_not * res^2

            mcd64a1_int = mcd64a1_int * res^2
            mcd64a1_have = mcd64a1_have * res^2
            mcd64a1_not = mcd64a1_not * res^2

            #project shapefiles
            ak_proj = ak %>% st_transform(crs(fire_cci, proj = T))
            ca_proj = ca %>% st_transform(crs(fire_cci, proj = T))

            #extract ak
            # fire_cci_ak = crop(fire_cci, vect(ak_proj))
            fire_cci_ak = mask(fire_cci, vect(ak_proj))
            fire_cci_ak_int = mask(fire_cci_int, vect(ak_proj))
            fire_cci_ak_have = mask(fire_cci_have, vect(ak_proj))
            fire_cci_ak_not = mask(fire_cci_not, vect(ak_proj))


            # mcd64a1_ak = crop(mcd64a1, vect(ak_proj))
            mcd64a1_ak = mask(mcd64a1, vect(ak_proj))
            mcd64a1_ak_int = mask(mcd64a1_int, vect(ak_proj))
            mcd64a1_ak_have = mask(mcd64a1_have, vect(ak_proj))
            mcd64a1_ak_not = mask(mcd64a1_not, vect(ak_proj))

            # lfdb_ak = crop(lfdb, vect(ak_proj))
            lfdb_ak = mask(lfdb, vect(ak_proj))


            #extract ca
            # fire_cci_ca = crop(fire_cci, vect(ca_proj))
            fire_cci_ca = mask(fire_cci, vect(ca_proj))
            fire_cci_ca_int = mask(fire_cci_int, vect(ca_proj))
            fire_cci_ca_have = mask(fire_cci_have, vect(ca_proj))
            fire_cci_ca_not = mask(fire_cci_not, vect(ca_proj))

            # mcd64a1_ca = crop(mcd64a1, vect(ca_proj))
            mcd64a1_ca = mask(mcd64a1, vect(ca_proj))
            mcd64a1_ca_int = mask(mcd64a1_int, vect(ca_proj))
            mcd64a1_ca_have = mask(mcd64a1_have, vect(ca_proj))
            mcd64a1_ca_not = mask(mcd64a1_not, vect(ca_proj))    

            # lfdb_ca = crop(lfdb, vect(ca_proj))
            lfdb_ca = mask(lfdb, vect(ca_proj))


            #get all values in tibble
            fire_cci_ak = tibble(Area = values(fire_cci_ak, mat = F)) %>% drop_na()
            fire_cci_ak = fire_cci_ak %>% mutate(Class = 'Us', AOI = 'AK')

            fire_cci_ak_int = tibble(Area = values(fire_cci_ak_int, mat = F)) %>% drop_na()
            fire_cci_ak_int = fire_cci_ak_int %>% mutate(Class = 'Us-Intersection', AOI = 'AK')

            fire_cci_ak_have = tibble(Area = values(fire_cci_ak_have, mat = F)) %>% drop_na()
            fire_cci_ak_have = fire_cci_ak_have %>% mutate(Class = 'Us-Have', AOI = 'AK')

            fire_cci_ak_not = tibble(Area = values(fire_cci_ak_not, mat = F)) %>% drop_na()
            fire_cci_ak_not = fire_cci_ak_not %>% mutate(Class = 'Us-Not', AOI = 'AK')

            fire_cci_ca = tibble(Area = values(fire_cci_ca, mat = F)) %>% drop_na()
            fire_cci_ca = fire_cci_ca %>% mutate(Class = 'Us', AOI = 'CA')

            fire_cci_ca_int = tibble(Area = values(fire_cci_ca_int, mat = F)) %>% drop_na()
            fire_cci_ca_int = fire_cci_ca_int %>% mutate(Class = 'Us-Intersection', AOI = 'CA')

            fire_cci_ca_have = tibble(Area = values(fire_cci_ca_have, mat = F)) %>% drop_na()
            fire_cci_ca_have = fire_cci_ca_have %>% mutate(Class = 'Us-Have', AOI = 'CA')

            fire_cci_ca_not = tibble(Area = values(fire_cci_ca_not, mat = F)) %>% drop_na()
            fire_cci_ca_not = fire_cci_ca_not %>% mutate(Class = 'Us-Not', AOI = 'CA')

            mcd64a1_ak = tibble(Area = values(mcd64a1_ak, mat = F)) %>% drop_na()
            mcd64a1_ak = mcd64a1_ak %>% mutate(Class = 'GABAM', AOI = 'AK')

            mcd64a1_ak_int = tibble(Area = values(mcd64a1_ak_int, mat = F)) %>% drop_na()
            mcd64a1_ak_int = mcd64a1_ak_int %>% mutate(Class = 'GABAM-Intersection', AOI = 'AK')

            mcd64a1_ak_have = tibble(Area = values(mcd64a1_ak_have, mat = F)) %>% drop_na()
            mcd64a1_ak_have = mcd64a1_ak_have %>% mutate(Class = 'GABAM-Have', AOI = 'AK')

            mcd64a1_ak_not = tibble(Area = values(mcd64a1_ak_not, mat = F)) %>% drop_na()
            mcd64a1_ak_not = mcd64a1_ak_not %>% mutate(Class = 'GABAM-Not', AOI = 'AK')


            mcd64a1_ca = tibble(Area = values(mcd64a1_ca, mat = F)) %>% drop_na()
            mcd64a1_ca = mcd64a1_ca %>% mutate(Class = 'GABAM', AOI = 'CA')

            mcd64a1_ca_int = tibble(Area = values(mcd64a1_ca_int, mat = F)) %>% drop_na()
            mcd64a1_ca_int = mcd64a1_ca_int %>% mutate(Class = 'GABAM-Intersection', AOI = 'CA')

            mcd64a1_ca_have = tibble(Area = values(mcd64a1_ca_have, mat = F)) %>% drop_na()
            mcd64a1_ca_have = mcd64a1_ca_have %>% mutate(Class = 'GABAM-Have', AOI = 'CA')

            mcd64a1_ca_not = tibble(Area = values(mcd64a1_ca_not, mat = F)) %>% drop_na()
            mcd64a1_ca_not = mcd64a1_ca_not %>% mutate(Class = 'GABAM-Not', AOI = 'CA')


            lfdb_ak = tibble(Area = values(lfdb_ak, mat = F)) %>% drop_na()
            lfdb_ak = lfdb_ak %>% mutate(Class = 'MTBS', AOI = 'AK')

            lfdb_ca = tibble(Area = values(lfdb_ca, mat = F)) %>% drop_na()
            lfdb_ca = lfdb_ca %>% mutate(Class = 'NBAC', AOI = 'CA')



            #combine
            final = bind_rows(list(fire_cci_ak, 
                                   fire_cci_ca, 
                                   mcd64a1_ak, 
                                   mcd64a1_ca, 
                                   lfdb_ak, 
                                   lfdb_ca, 
                                   fire_cci_ak_int, 
                                   fire_cci_ak_have, 
                                   fire_cci_ak_not,
                                   fire_cci_ca_int, 
                                   fire_cci_ca_have, 
                                   fire_cci_ca_not,
                                   mcd64a1_ak_int, 
                                   mcd64a1_ak_have, 
                                   mcd64a1_ak_not,
                                   mcd64a1_ca_int, 
                                   mcd64a1_ca_have, 
                                   mcd64a1_ca_not))
            final$Year = year

            new_f = str_replace(f, '.tif', '.csv')
            write_csv(final, file.path(out_path, new_f))

            # combined[[length(combined) + 1]] = final

            print(new_f)     
        }
    }
    
}       
# combined = bind_rows(combined)
# write_csv(combined, file.path(out_path, 'landsat.csv'))


't'


# fire_cci_ak_int




