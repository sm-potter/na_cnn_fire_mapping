import os
import ee
import geemap
from geeml.extract import extractor
from google.cloud import storage
from google.cloud import client
import random


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/explore/nobackup/people/spotter5/cnn_mapping/gee-serdp-upload-7cd81da3dc69.json"

service_account = 'gee-serdp-upload@appspot.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, "/explore/nobackup/people/spotter5/cnn_mapping/gee-serdp-upload-7cd81da3dc69.json")
ee.Initialize(credentials)
# Initialize GEE with high-volume end-point
# ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
ee.Initialize()

sent_2A = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") #sentinel 2
s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') #cloud masking for sentinel
lfdb = ee.FeatureCollection("users/spotter/fire_cnn/raw/nbac_1985") #nbac_fire_polygons, this can be any polygon shapefile, final version would be nbac and mtbs, this is nbac polygons
# lfdb = ee.FeatureCollection("users/spotter/fire_cnn/raw/ak_mtbs_1985") #nbac_fire_polygons, this can be any polygon shapefile, final version would be nbac and mtbs, this is mtbs polygons


#probability of clouds
MAX_CLOUD_PROBABILITY = 50

def sent_maskcloud(image):
    
    
    image = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])# rename bands to match landsat
  
    image =  image.toShort()
    
    clouds = ee.Image(image.get('cloud_mask')).select('probability')
    
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    
    image = image.updateMask(isNotCloud)

    #reproject 30m but remember b1, b2 and b3 are 10 and the rest are 20
    image1 = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4'])
    image2 = image.select(['SR_B5', 'SR_B7'])

    
    image1 = image1.reproject(
    crs = image1.projection().crs(),
    scale = 30) #resample for landsat
    
    
    image2 = image2.reproject(
    crs = image2.projection().crs(),
    scale = 30) #resample for landsat
    
    image = image1.addBands(image2)
    
    return image 

#Join S2 SR with cloud probability dataset to add cloud mask.
s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(
    
  primary=sent_2A,
  secondary=s2Clouds,
  condition=ee.Filter.equals(leftField='system:index', rightField='system:index'))

#apply cloud masking
sent_2A = ee.ImageCollection(s2SrWithCloudMask).map(sent_maskcloud)


def mask(image):
    qa = image.select('QA_PIXEL')                                       
    mask = qa.bitwiseAnd(8).eq(0).And(qa.bitwiseAnd(10).eq(0)).And(qa.bitwiseAnd(32).eq(0))  
    return(image.updateMask(mask))

def land_scale(image):

    return(image.multiply(0.0000275).add(-0.2))

def sent_scale(image):
    return(image.multiply(0.0001))

import pandas as pd
coeffs = pd.read_csv("/explore/nobackup/people/spotter5/cnn_mapping/raw_files/boreal_xcal_regression_coefficients.csv").fillna(0)

#l5
def landsat_correct(sat, bands):

    """argument 1 is which sattelite, LANDASAT_5 or LANDSAT_8
    argument 2 is bands of interest.  Bands must be in same order as EE,
    
    regression is of form,
    L7 = B0 + (B1 * L5/8) + (B2 * L^2) + (B3 * L^3)
    """

    #bands of interest in order of interest
    l5 = coeffs[(coeffs['satellite'] == sat) & (coeffs['band.or.si'] .isin (bands))] 

    #arrange the band or si column
    l5['band.or.si']=pd.Categorical(l5['band.or.si'],categories=bands)
    l5=l5.sort_values('band.or.si')

    b0 = l5['B0'].values.tolist()
    b1 = l5['B1'].values.tolist()
    b2 = l5['B2'].values.tolist()
    b3 = l5['B3'].values.tolist()

    return (b0, b1, b2, b3)

#get the corrections, each output is a list at one of the four locations
l8_corr = landsat_correct(sat = 'LANDSAT_8', bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'nbr', 'ndvi', 'ndii'])
l5_corr = landsat_correct(sat = 'LANDSAT_5', bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'nbr', 'ndvi', 'ndii'])

os.environ["GCLOUD_PROJECT"] = "gee-serdp-upload"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/explore/nobackup/people/spotter5/cnn_mapping/gee-serdp-upload-7cd81da3dc69.json"
storage_client = storage.Client.from_service_account_json("/explore/nobackup/people/spotter5/cnn_mapping/gee-serdp-upload-7cd81da3dc69.json")

os.environ["GCLOUD_PROJECT"] = "gee-serdp-upload"
storage_client = storage.Client()
# bucket_name = 'smp-scratch/mtbs_1985'
bucket_name = 'smp-scratch'

bucket = storage_client.bucket(bucket_name)

def to_float(image):

    b1 = image.select('SR_B1').cast({'SR_B1':'float'}) #0
    b2 = image.select('SR_B2').cast({'SR_B2':'float'}) #1
    b3 = image.select('SR_B3').cast({'SR_B3':'float'}) #2
    b4 = image.select('SR_B4').cast({'SR_B4':'float'}) #3
    b5 = image.select('SR_B5').cast({'SR_B5':'float'}) #4
    b6 = image.select('SR_B7').cast({'SR_B7':'float'}) #5

    image = b1.addBands(b2).addBands(b3).addBands(b4).addBands(b5).addBands(b6)

    return image


#get all the ids within the lfdb shapefile
all_ids = ee.List(lfdb.distinct(["ID"]).aggregate_array("ID"))
all_ids = all_ids.getInfo()

# Specify the folder within your bucket
folder_name = 'nbac_old'

#loop through each fire polygon
for i in all_ids:

    #name of output file
    # fname = f"median_{i}.tif"
    fname = f"{folder_name}/median_{i}.tif"

    fname2 = f"{folder_name}/median_{i}.tif.tif"

    #check if file exists on my bucket, if it does skip
    stats = storage.Blob(bucket=bucket, name=fname2).exists(storage_client)
    
    if stats:
        print(f"File {fname} already exists in the bucket. Skipping...")

    else:
        print(f"File {fname} does not exist")

    # #check if file exists on my bucket, if it does skip
    # stats = storage.Blob(bucket=bucket, name=fname).exists(storage_client)
    # if stats == False:
        
        #get the fire polygon of interest
        sub_shape = lfdb.filter(ee.Filter.eq("ID", i))

        #get all other fire ids that are not this one
        not_fires = lfdb.filter(ee.Filter.neq("ID", i))

        #get the year of this fire
        this_year = ee.Number(sub_shape.aggregate_array('Year').get(0))
      
        #date ranges for pre and post fire
        pre_start = ee.Date.fromYMD(this_year.subtract(1), 6, 1)
        pre_end = ee.Date.fromYMD(this_year.subtract(1), 8, 31)
        post_start = pre_start.advance(2, 'year')
        post_end = pre_end.advance(2, 'year')


        #first get the bounding box of the fire
        bbox = sub_shape.geometry().bounds()


        #offset the bounding box by a random number
        # all_rands = [0.00, 0.02, -0.02]
        
        all_rands = [0.00]

        rand1 = random.sample(all_rands, 1)[0]
        rand2 = random.sample(all_rands, 1)[0]

        #offset applied
        proj = ee.Projection("EPSG:4326").translate(rand1, rand2)
        
        #for the bounding box apply the randomly selected offset
        final_buffer = ee.Geometry.Polygon(bbox.coordinates(), proj).transform(proj)
        
        #this is a bit of a hack but we have two different bounding box sizes because when we export we need to use some additonal area to avoid cuttoffs
        final_buffer2 = final_buffer.buffer(distance= 5000).bounds()

        final_buffer = final_buffer.buffer(distance= 40000)#.bounds().transform(proj='EPSG:3413', maxError=1)
    
     
        #just getting some date info here to ensure pre fire is one  year before and post fire is one year after the fire year of interest
        startYear = pre_start.get('year')

        #convert to client side
        startYear = startYear.getInfo()  # local string
        endYear = str(int(startYear) + 2)
        startYear = str(startYear)

        #------------------------------------------------harmonize landsat
        startDay  = '01-01' # what is the beginning of date filter | month-day
        endDay     = '12-30' # what is the end of date filter | month-day



        #------ RETRIEVE A SENSOR SR COLLECTION FUNCTION -----
        def getSRcollection(start_date, end_date, sensor):
            # get a landsat collection for given year, day range, and sensor
            srCollection = ee.ImageCollection('LANDSAT/'+ sensor + '/C02/T1_L2').filterDate(start_date, end_date)

            return srCollection

        #get the image collections only in the dates of interest and filter by the final buffer aoi
        lt5 = getSRcollection(startYear+'-'+startDay, endYear+'-'+endDay, 'LT05').filterBounds(final_buffer)    
        le7 = getSRcollection(startYear+'-'+startDay, endYear+'-'+endDay, 'LE07').filterBounds(final_buffer)     
        lc8 = getSRcollection(startYear+'-'+startDay, endYear+'-'+endDay, 'LC08').filterBounds(final_buffer)        
        sent= sent_2A.filterBounds(final_buffer)

        #------------------------------------------Landsat 5 corrections

        #pre fire apply landsat mask, scale factor etc.
        pre_lt5 = lt5.filterDate(pre_start, pre_end).map(mask).map(land_scale).filterBounds(final_buffer).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).map(to_float)

        #ensure we have imagery for the sensor
        if pre_lt5.size().getInfo() > 0 :


            #take the median pf the collection
            pre_lt5 = pre_lt5.median().clip(final_buffer)

            #calculate nbr, ndvi and ndii
            pre_lt5_nbr = pre_lt5.normalizedDifference(['SR_B4', 'SR_B7']).select([0], ['NBR']).toFloat()
            pre_lt5_ndvi = pre_lt5.normalizedDifference(['SR_B4', 'SR_B3']).select([0], ['NDVI']).toFloat()
            pre_lt5_ndii = pre_lt5.normalizedDifference(['SR_B4', 'SR_B5']).select([0], ['NDII']).toFloat()

            #add the bands back
            pre_lt5 = pre_lt5.addBands(pre_lt5_nbr).addBands(pre_lt5_ndvi).addBands(pre_lt5_ndii)

            #apply the corrections from Logan
            l5_pre_corr = pre_lt5.multiply(l5_corr[1]).add(pre_lt5.pow(2).multiply(l5_corr[2])).add(pre_lt5.pow(3).multiply(l5_corr[3])).add(l5_corr[0])

        #-------now do post-fire
        #select bands
        post_lt5 = lt5.filterDate(post_start, post_end).map(mask).map(land_scale).filterBounds(final_buffer).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).map(to_float)

        #       #ensure we have imagery for the sensor
        if post_lt5.size().getInfo() > 0 :

            #take the median
            post_lt5 = post_lt5.median().clip(final_buffer)

            #calculate nbr, ndvi and ndii
            post_lt5_nbr = post_lt5.normalizedDifference(['SR_B4', 'SR_B7']).select([0], ['NBR']).toFloat()
            post_lt5_ndvi = post_lt5.normalizedDifference(['SR_B4', 'SR_B3']).select([0], ['NDVI']).toFloat()
            post_lt5_ndii = post_lt5.normalizedDifference(['SR_B4', 'SR_B5']).select([0], ['NDII']).toFloat()

            #add the bands back
            post_lt5 = post_lt5.addBands(post_lt5_nbr).addBands(post_lt5_ndvi).addBands(post_lt5_ndii)

            #apply the corrections

            l5_post_corr = post_lt5.multiply(l5_corr[1]).add(post_lt5.pow(2).multiply(l5_corr[2])).add(post_lt5.pow(3).multiply(l5_corr[3])).add(l5_corr[0])


        #------------------------------------------Landsat 7, no corrections but get things clipped and do pre fire/post_fire stuff


        #select bands
        pre_le7 = le7.filterDate(pre_start, pre_end).map(mask).map(land_scale).filterBounds(final_buffer).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).map(to_float)

              #ensure we have imagery for the sensor
        if pre_le7.size().getInfo() > 0 :

            #take the median
            pre_le7 = pre_le7.median().clip(final_buffer)

            #calculate nbr, ndvi and ndii
            pre_le7_nbr = pre_le7.normalizedDifference(['SR_B4', 'SR_B7']).select([0], ['NBR']).toFloat()
            pre_le7_ndvi = pre_le7.normalizedDifference(['SR_B4', 'SR_B3']).select([0], ['NDVI']).toFloat()
            pre_le7_ndii = pre_le7.normalizedDifference(['SR_B4', 'SR_B5']).select([0], ['NDII']).toFloat()

            #add the bands back
            pre_le72 = pre_le7.addBands(pre_le7_nbr).addBands(pre_le7_ndvi).addBands(pre_le7_ndii)

#         #-------now do post-fire
#         #select bands
        post_le7 = le7.filterDate(post_start, post_end).map(mask).map(land_scale).filterBounds(final_buffer).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).map(to_float)

        #       #ensure we have imagery for the sensor
        if post_le7.size().getInfo() > 0 :


            #take the median
            post_le7 = post_le7.median().clip(final_buffer)

            #calculate nbr, ndvi and ndii
            post_le7_nbr = post_le7.normalizedDifference(['SR_B4', 'SR_B7']).select([0], ['NBR']).toFloat()
            post_le7_ndvi = post_le7.normalizedDifference(['SR_B4', 'SR_B3']).select([0], ['NDVI']).toFloat()
            post_le7_ndii = post_le7.normalizedDifference(['SR_B4', 'SR_B5']).select([0], ['NDII']).toFloat()

            #add the bands back
            post_le72 = post_le7.addBands(post_le7_nbr).addBands(post_le7_ndvi).addBands(post_le7_ndii)

        #------------------------------------------Landsat 8 corrections


        #-------first do pre-fire

        #select bands, note here we rename to match landsat 5 and 7
        pre_lc8 = lc8.filterDate(pre_start, pre_end).map(mask).map(land_scale).filterBounds(final_buffer).select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).map(to_float)

        #       #ensure we have imagery for the sensor
        if pre_lc8.size().getInfo() > 0 :

            #take the median
            pre_lc8 = pre_lc8.median().clip(final_buffer)

            #calculate nbr, ndvi and ndii
            pre_lc8_nbr = pre_lc8.normalizedDifference(['SR_B4', 'SR_B7']).select([0], ['NBR']).toFloat()
            pre_lc8_ndvi = pre_lc8.normalizedDifference(['SR_B4', 'SR_B3']).select([0], ['NDVI']).toFloat()
            pre_lc8_ndii = pre_lc8.normalizedDifference(['SR_B4', 'SR_B5']).select([0], ['NDII']).toFloat()

            #add the bands back
            pre_lc8 = pre_lc8.addBands(pre_lc8_nbr).addBands(pre_lc8_ndvi).addBands(pre_lc8_ndii)

            #apply the corrections
            l8_pre_corr = pre_lc8.multiply(l8_corr[1]).add(pre_lc8.pow(2).multiply(l8_corr[2])).add(pre_lc8.pow(3).multiply(l8_corr[3])).add(l8_corr[0])

        #-------now do post-fire
          #select bands
        post_lc8 = lc8.filterDate(post_start, post_end).map(mask).map(land_scale).filterBounds(final_buffer).select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).map(to_float)

        #       #ensure we have imagery for the sensor
        if post_lc8.size().getInfo() > 0 :

        #take the median
            post_lc8 = post_lc8.median().clip(final_buffer)

            #calculate nbr, ndvi and ndii
            post_lc8_nbr = post_lc8.normalizedDifference(['SR_B4', 'SR_B7']).select([0], ['NBR']).toFloat()
            post_lc8_ndvi = post_lc8.normalizedDifference(['SR_B4', 'SR_B3']).select([0], ['NDVI']).toFloat()
            post_lc8_ndii = post_lc8.normalizedDifference(['SR_B4', 'SR_B5']).select([0], ['NDII']).toFloat()

            #add the bands back
            post_lc8 = post_lc8.addBands(post_lc8_nbr).addBands(post_lc8_ndvi).addBands(post_lc8_ndii)

            #apply the corrections

            l8_post_corr = post_lc8.multiply(l8_corr[1]).add(post_lc8.pow(2).multiply(l8_corr[2])).add(post_lc8.pow(3).multiply(l8_corr[3])).add(l8_corr[0])
        #------------------------------------------Sentinel 2 corrections, use landsat 8 coefficients


        #-------first do pre-fire

        # select bands
        pre_sent = sent_2A.filterDate(pre_start, pre_end).map(sent_scale).filterBounds(final_buffer).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).map(to_float)

        #       #ensure we have imagery for the sensor
        if pre_sent.size().getInfo() > 0 :

            #take the median
            pre_sent = pre_sent.median().clip(final_buffer)

            #calculate nbr, ndvi and ndii
            pre_sent_nbr = pre_sent.normalizedDifference(['SR_B4', 'SR_B7']).select([0], ['NBR']).toFloat()
            pre_sent_ndvi = pre_sent.normalizedDifference(['SR_B4', 'SR_B3']).select([0], ['NDVI']).toFloat()
            pre_sent_ndii = pre_sent.normalizedDifference(['SR_B4', 'SR_B5']).select([0], ['NDII']).toFloat()

            #add the bands back
            pre_sent = pre_sent.addBands(pre_sent_nbr).addBands(pre_sent_ndvi).addBands(pre_sent_ndii)

            #apply the corrections

            sent_pre_corr = pre_sent.multiply(l8_corr[1]).add(pre_sent.pow(2).multiply(l8_corr[2])).add(pre_sent.pow(3).multiply(l8_corr[3])).add(l8_corr[0])

        #-------now do post-fire
        #select bands
        post_sent = sent_2A.filterDate(post_start, post_end).map(sent_scale).filterBounds(final_buffer).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).map(to_float)

        #       #ensure we have imagery for the sensor
        if post_sent.size().getInfo() > 0 :

            #take the median
            post_sent = post_sent.median().clip(final_buffer)

            #calculate nbr, ndvi and ndii
            post_sent_nbr = post_sent.normalizedDifference(['SR_B4', 'SR_B7']).select([0], ['NBR']).toFloat()
            post_sent_ndvi = post_sent.normalizedDifference(['SR_B4', 'SR_B3']).select([0], ['NDVI']).toFloat()
            post_sent_ndii = post_sent.normalizedDifference(['SR_B4', 'SR_B5']).select([0], ['NDII']).toFloat()

            #add the bands back
            post_sent = post_sent.addBands(post_sent_nbr).addBands(post_sent_ndvi).addBands(post_sent_ndii)

            #apply the corrections

            sent_post_corr = post_sent.multiply(l8_corr[1]).add(post_sent.pow(2).multiply(l8_corr[2])).add(post_sent.pow(3).multiply(l8_corr[3])).add(l8_corr[0])


        #try to see if image exists, if so append

        #----------------------all prefire

        #empty list for pre-fire, use this to combine if we have land 5, 7, 8 or sentinel 
        pre_input = []

        try:
            l5_pre_corr.getInfo()
            pre_input.append(l5_pre_corr)

        except:
            pass

        try:
            pre_le72.getInfo()
            pre_input.append(pre_le72)

        except:
            pass

        try:
            l8_pre_corr.getInfo()
            pre_input.append(l8_pre_corr)

        except:
            pass

        try:
            sent_pre_corr.getInfo()
            pre_input.append(sent_pre_corr)

        except:
            pass


        #----------------------all postfire

        #empty list for post-fire, use this to combine if we have land 5, 7, 8 or sentinel 
        post_input = []

        try:
            l5_post_corr.getInfo()
            post_input.append(l5_post_corr)

        except:
            pass

        try:
            post_le72.getInfo()
            post_input.append(post_le72)

        except:
            pass

        try:
            l8_post_corr.getInfo()
            post_input.append(l8_post_corr)

        except:
            pass

        try:
            sent_post_corr.getInfo()
            post_input.append(sent_post_corr)

        except:
            pass
    
        
        #ensure we have both pre fire and post fire imagery, otherwise can't get training data
        if (len(pre_input) >0) and (len(post_input) > 0):

            #take the median of the image collections
            pre_input = ee.ImageCollection(pre_input)
            post_input = ee.ImageCollection(post_input)

            pre_input = pre_input.median()
            post_input= post_input.median()

            #difference the bands fore pre and post fire
            raw_bands = pre_input.subtract(post_input).multiply(1000)

            #we need to see which image ids from the entire lfdb are already included in the buffer
            lfdb_filtered_orig = lfdb.filterBounds(final_buffer)

            #ensure all fires are within the actual year of interest (this_year) and two years prior, otherwise ignore, this is to ensure we don't have nearby fires from previous years
            first_year =  int(startYear) + 1
            second_year =  int(startYear)
            third_year =  int(startYear) - 1
            fourth_year = int(startYear) + 2

            lfdb_filtered = lfdb_filtered_orig.filter(ee.Filter.eq("Year", first_year))

            bad_filtered = lfdb_filtered_orig.filter(ee.Filter.Or(ee.Filter.eq("Year", second_year), ee.Filter.eq("Year", third_year), ee.Filter.eq("Year", fourth_year)))


            #get ids which are in image
            all_ids_new = ee.List(lfdb_filtered.distinct(["ID"]).aggregate_array("ID")).getInfo()


            #remove ids from all dates which we do not need anymore
            all_ids2 = [i for i in all_ids if i not in all_ids_new]

            #area we have good fires
            fire_rast = lfdb_filtered.reduceToImage(properties= ['ID'], reducer = ee.Reducer.first())

            #areas we have fires from other years or nearby we don't want to use
            bad_fire_rast = bad_filtered.reduceToImage(properties= ['ID'], reducer = ee.Reducer.first())

            #change values to 1 for fire of interest
            fire_rast = fire_rast.where(fire_rast.gt(0), 1)

            #change values for bad fire raster to 1 as well
            bad_fire_rast = bad_fire_rast.where(bad_fire_rast.gt(0), 1)

            #if the fires overlap we want to keep those locations
            bad_fire_rast = bad_fire_rast.where(bad_fire_rast.eq(1).And(fire_rast.eq(1)), 2).unmask(-999)

            #rename to y for the fire raster
            fire_rast = fire_rast.rename(['y'])

            #copy the first values of raw_bands
            y = post_input.select(['SR_B2'], ['y'])

            #turn all values of y to 0
            y  = y.where(y.gt(0), 0)

            #turn values to 1 where fire_rast is 1
            y  = y.where(fire_rast.eq(1), 1)
            
            #we need to ensure all bands are shorts 
            b1 = raw_bands.select('SR_B1').cast({'SR_B1':'short'}) 
            b2 = raw_bands.select('SR_B2').cast({'SR_B2':'short'}) 
            b3 = raw_bands.select('SR_B3').cast({'SR_B3':'short'}) 
            b4 = raw_bands.select('SR_B4').cast({'SR_B4':'short'}) 
            b5 = raw_bands.select('SR_B5').cast({'SR_B5':'short'}) 
            b6 = raw_bands.select('SR_B7').cast({'SR_B7':'short'}) 
            b7 = raw_bands.select('NBR').cast({'NBR':'short'}) 
            b8 = raw_bands.select('NDVI').cast({'NDVI':'short'}) 
            b9 = raw_bands.select('NDII').cast({'NDII':'short'}) 
            b10 = y.select('y').cast({'y':'short'})
            
            #combine all the bands for predictors
            # raw_bands = b1.addBands(b2).addBands(b3).addBands(b4).addBands(b5).addBands(b6).addBands(b7).addBands(b8).addBands(b9)
            raw_bands = b7.addBands(b8).addBands(b9)

            
            #for areas where there are nearby fires or fires in previous years we set those to 0
            raw_bands = raw_bands.updateMask(bad_fire_rast.neq(1))
            
            #add in the target variable
            raw_bands = raw_bands.addBands(b10)

            #start download
            print(f"Downloading {fname}")
            

            #export image to my cloud storage
            task = ee.batch.Export.image.toCloudStorage(
                                    image = raw_bands.toShort(),
                                    region=final_buffer2, 
                                    description='median_' + str(i),
                                    scale=30,
                                    crs='EPSG:3413',
                                    maxPixels=1e13,
                                    bucket = 'smp-scratch',
                                    fileNamePrefix=fname)

            task.start()

