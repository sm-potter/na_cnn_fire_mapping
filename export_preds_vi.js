// this is an earth engine script located at /users/fire_cnn/export_preds_vi.  It will take the full AK/CA predictions from predict_model_domain_VI.ipynb and export locally while 
//masking out crops, whater, and only 1km buffered polygons, VIIRS active fire and MODIS actrive fire.  It will also export fire cci, MCD64A1, the reference polygons (as raster), GABAM and MCD64A1 so I //can compare burned areas for the paper

var ak = ee.FeatureCollection("users/spotter/alaska")
var mod_burn = ee.ImageCollection("MODIS/061/MCD64A1") //modis fire product
fire_cci = ee.ImageCollection("ESA/CCI/FireCCI/5_1"); //fire cci for comparison
var geometry = ee.Geometry.Polygon(
        [[[-169.37968750000002, 72.61861047682912],
          [-169.37968750000002, 40.20824456535776],
          [-50.55156250000001, 40.20824456535776],
          [-50.55156250000001, 72.61861047682912]]], null, false),
    
    
var dinerstein = ee.FeatureCollection("users/spotter/fire_cnn/raw/boreal_tundra"); //area to clip to which is boreal and tundra as defined by dinerstein ecoregions

//year of interest to export
var year = 2004

//fire cci 
var fire_cci = fire_cci.filterDate(String(year) + '-01-01', String(year) + '-12-31')

var filter_cl = function(image){
  
  var cl = image.select('ConfidenceLevel')
  image = image.updateMask(cl.gte(50)).select('BurnDate')
  return image
  
}

//MCD64A1
var mod_burn = mod_burn.filterDate(String(year) + '-01-01', String(year) + '-12-31').max().select('BurnDate')

fire_cci = fire_cci.map(filter_cl).max()

var lfdb = ee.FeatureCollection("users/spotter/fire_cnn/raw/ak_ca_1985").filter(ee.Filter.eq('Year',  year))   
var table = ee.FeatureCollection("users/spotter/fire_cnn/raw/ak_grid_gabam")


var ak = ee.FeatureCollection("users/spotter/alaska")
var sk = ee.FeatureCollection("users/spotter/SK_boundary")
var buffer = ee.FeatureCollection("users/spotter/fire_cnn/raw/2001_2022_ak_ca_buff").filter(ee.Filter.eq('Year',  year))   
var water = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
var grd = ee.FeatureCollection("users/spotter/fire_cnn/raw/ak_ca_grd_west_largest2")
var territory22 = ee.FeatureCollection("users/spotter/fire_cnn/raw/ak_ca_grd_west_largest2")

var water = water.filterDate(String(year) + '-01-01', String(year) + '-12-31').max().select('waterClass')//.clip(dinerstein)
var water = water.updateMask(water.eq(3))

//modis land cover for now
// var lc = ee.ImageCollection("MODIS/061/MCD12Q1")
//clip lc
// lc = lc.filterDate(String(year) + '-01-01', String(year) + '-12-31').max().clip(table.union())
// lc = lc.select('LC_Type1')

//cropland masking layyers
var crop_one = ee.ImageCollection('users/potapovpeter/Global_cropland_2003').max().set('system:time_start', '2001-01-01')
var crop_two = ee.ImageCollection('users/potapovpeter/Global_cropland_2003').max().set('system:time_start', '2002-01-01')
var crop_three = ee.ImageCollection('users/potapovpeter/Global_cropland_2003').max().set('system:time_start', '2003-01-01')
var crop_four = ee.ImageCollection('users/potapovpeter/Global_cropland_2007').max().set('system:time_start', '2004-01-01')
var crop_five = ee.ImageCollection('users/potapovpeter/Global_cropland_2007').max().set('system:time_start', '2005-01-01')
var crop_six = ee.ImageCollection('users/potapovpeter/Global_cropland_2007').max().set('system:time_start', '2006-01-01')
var crop_seven = ee.ImageCollection('users/potapovpeter/Global_cropland_2007').max().set('system:time_start', '2007-01-01')
var crop_eight = ee.ImageCollection('users/potapovpeter/Global_cropland_2011').max().set('system:time_start', '2008-01-01')
var crop_nine = ee.ImageCollection('users/potapovpeter/Global_cropland_2011').max().set('system:time_start', '2009-01-01')
var crop_ten = ee.ImageCollection('users/potapovpeter/Global_cropland_2011').max().set('system:time_start', '2010-01-01')
var crop_eleven = ee.ImageCollection('users/potapovpeter/Global_cropland_2011').max().set('system:time_start', '2011-01-01')
var crop_twelve = ee.ImageCollection('users/potapovpeter/Global_cropland_2015').max().set('system:time_start', '2012-01-01')
var crop_thirteen = ee.ImageCollection('users/potapovpeter/Global_cropland_2015').max().set('system:time_start', '2013-01-01')
var crop_fourteen = ee.ImageCollection('users/potapovpeter/Global_cropland_2015').max().set('system:time_start', '2014-01-01')
var crop_fifteen = ee.ImageCollection('users/potapovpeter/Global_cropland_2015').max().set('system:time_start', '2015-01-01')
var crop_sixteen = ee.ImageCollection('users/potapovpeter/Global_cropland_2019').max().set('system:time_start', '2016-01-01')
var crop_seventeen = ee.ImageCollection('users/potapovpeter/Global_cropland_2019').max().set('system:time_start', '2017-01-01')
var crop_eighteen = ee.ImageCollection('users/potapovpeter/Global_cropland_2019').max().set('system:time_start', '2018-01-01')
var crop_nineteen = ee.ImageCollection('users/potapovpeter/Global_cropland_2019').max().set('system:time_start', '2019-01-01')
var crop_twenty = ee.ImageCollection('users/potapovpeter/Global_cropland_2019').max().set('system:time_start', '2020-01-01')

var crop = ee.ImageCollection([crop_one, crop_two,
                              crop_three, crop_four,
                              crop_five, crop_six,
                              crop_seven, crop_eight,
                              crop_nine, crop_ten,
                              crop_eleven, crop_twelve,
                              crop_thirteen, crop_fourteen,
                              crop_fifteen, crop_sixteen,
                              crop_seventeen, crop_eighteen,
                              crop_nineteen, crop_twenty]).filterDate(String(year) + '-01-01', String(year) + '-12-31').max()
                               

//read in gabam which is a 30m random forest model
var gabam = ee.ImageCollection("projects/sat-io/open-datasets/GABAM")

gabam = gabam.filterDate(String(year) + '-01-01', String(year) + '-12-31').mean()




//union to a single feature for the table
var aoi =dinerstein//.union()



//get modis active fire and merge with viirs
var Terra_Fire = ee.ImageCollection("MODIS/061/MOD14A1").filterBounds(aoi)
var Aqua_Fire = ee.ImageCollection("MODIS/061/MYD14A1").filterBounds(aoi)


var mod_fire = Terra_Fire.merge(Aqua_Fire).filterBounds(aoi).filterDate(String(year) + '-01-01', String(year) + '-12-31')

var add_bands = function(image){
    image = image.addBands(ee.Image(ee.Date(image.get('system:time_start')).getRelative('day','year').add(1)).clamp(1,366)).updateMask(image.select('FireMask').gte(7))
    return image.updateMask(image.select('constant').gt(60))
}
    
mod_fire =  mod_fire.map(add_bands)

mod_fire = mod_fire.select('constant').max().clip(aoi)


//now buffer the buffered 2015 fire polygons, turn it to an image and merge with modis and viirs
var to_year = function(f){
  return(f.set({'Year': year}))
};

buffer = buffer.map(to_year)

var buff_img = buffer.reduceToImage({
  properties: ['Year'],
  reducer: ee.Reducer.max()
}).clip(aoi).select(['max'], ['constant']);

var mod_lfdb = ee.ImageCollection([mod_fire.cast({'constant': 'short'}), buff_img.cast({'constant': 'short'})]).max()

Map.addLayer(mod_lfdb, {}, 'mod_lfdb_orig')
Map.addLayer(buff_img, {}, 'buff_img')
Map.addLayer(mod_lfdb, {}, 'mod_lfdb')

// read in viirs
var viirs = ee.Image("users/spotter/fire_cnn/VIIRS/" + String(year))
var viirs = viirs.clip(territory22).select(['b1'], ['constant']).cast({'constant': 'short'})

mod_lfdb = mod_lfdb.cast({'constant': 'short'})

//merge all
var final = ee.Image(ee.Algorithms.If(year <2012, mod_lfdb,  ee.ImageCollection([mod_lfdb, viirs]).max()))
final = final.cast({'constant': 'short'})



// read in the predictions and mask them with the areas
// var predictions = ee.ImageCollection("projects/gee-serdp-upload/assets/cnn_mapping/ak_ca" + String(year) + "_preds_128_32").max()
var predictions = ee.Image("users/spotter/fire_cnn/ak_ca_predictions/ak_ca_" + String(year) + "_preds_VI").clip(dinerstein).updateMask(water.unmask().not())
var predictions = ee.ImageCollection('projects/gee-serdp-upload/assets/cnn_mapping/ak_ca_' + String(year) + '_preds_VI').clip(dinerstein).updateMask(water.unmask().not())

Map.addLayer(predictions, {min: 0, max: 1000}, 'Raw Predictions')

Map.addLayer(final, {palette: 'blue'}, 'Constrained Prediction Locations')

//threshold at 0.5 probability
var thresholded = predictions.select('prediction').updateMask(final).updateMask(predictions.select('prediction').gte(500)).clip(aoi).updateMask(crop.neq(1))

//water mask
thresholded = thresholded.updateMask(water.unmask().not())
Map.addLayer(thresholded, {color: 'blue'}, 'Final Predictions')

//remove areas from gabam where I didn't have imagery
gabam = gabam.updateMask(predictions.gte(0))


gabam = gabam.updateMask(crop.neq(1))

//make image of fire databases
var lfdb_img = lfdb.reduceToImage({
  properties: ['Year'],
  reducer: ee.Reducer.max()
}).clip(aoi).select(['max'], ['constant']).updateMask(predictions.gte(0));

// Map.addLayer(thresholded)
var final2 = thresholded.addBands(gabam.toShort()).addBands(lfdb_img.toShort()).clip(dinerstein).updateMask(crop.neq(1))
// Map.addLayer(final2, {}, 'Final Predictions')


// var modis = fire_cci.toShort().addBands(mod_fire.toShort()).updateMask(predictions.gte(0)).updateMask(crop.neq(1))
// Map.addLayer(modis)

var fire_cci = fire_cci.toShort().updateMask(predictions.gte(0)).updateMask(crop.neq(1)).clip(dinerstein)

var mcd64a1 = mod_fire.toShort().updateMask(predictions.gte(0)).updateMask(crop.neq(1)).clip(dinerstein)
// Map.addLayer(fire_cci)
var image = ee.Image(String('users/spotter/Combustion_fires/Final_collection/2019_v5'));
var transform = image.projection().getInfo().transform;

Export.image.toCloudStorage({
  image: final2,
  description: 'predicted_' + String(year) + '_constrained_VI',
  bucket: 'smp-scratch',
  scale: 30,
  crs: 'EPSG:3413',
  // crsTransform: transform,
  maxPixels: 1e13,
  region : geometry,
  fileFormat: 'GeoTIFF'

});



// Export.image.toCloudStorage({
//   image: modis,
//   description: 'modis_ba_' + String(year),
//   bucket: 'smp-scratch',
//   // scale: 500,
//   crs:'SR-ORG:6974',
//   crsTransform:transform,
//   maxPixels: 1e13,
//   region : geometry,
//   fileFormat: 'GeoTIFF'

// });

Export.image.toCloudStorage({
  image: mcd64a1,
  description: 'mcd64a1_' + String(year),
  bucket: 'smp-scratch',
  scale: 463.312716,
  crs:'SR-ORG:6974',
  // crsTransform:transform,
  maxPixels: 1e13,
  region : geometry,
  fileFormat: 'GeoTIFF'

});

// Export.image.toCloudStorage({
//   image: fire_cci,
//   description: 'fire_cci_' + String(year),
//   bucket: 'smp-scratch',
//   scale: 250,
//   crs:'SR-ORG:6974',
//   // crsTransform:transform,
//   maxPixels: 1e13,
//   region : geometry,
//   fileFormat: 'GeoTIFF'

// });
  