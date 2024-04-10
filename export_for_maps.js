//this script will export predictions for individual fires 
//year of interest, can only be 2004, 2014, 2015
var year = 2015


//Imports
var utils = require('users/gena/packages:utils')
var palettes = require('users/gena/packages:palettes')

var dnbr_palette = palettes.colorbrewer.RdBu[9].reverse()
var dndvi_palette = palettes.colorbrewer.RdYlGn[9].reverse()
var dndii_palette = palettes.colorbrewer.PuOr[9].reverse()

var prediction_palette = palettes.colorbrewer.YlOrRd[9]


var vi = ee.Image('users/spotter/fire_cnn/ak_ca_VI/' + String(year))
var lfdb = ee.FeatureCollection("users/spotter/fire_cnn/raw/ak_ca_1985")
var table = geometry


var ak = ee.FeatureCollection("users/spotter/alaska")
var sk = ee.FeatureCollection("users/spotter/SK_boundary")
var buffer = ee.FeatureCollection("users/spotter/fire_cnn/raw/2001_2022_ak_ca_buff").filter(ee.Filter.eq('Year',  year))  
var water = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
var grd = ee.FeatureCollection("users/spotter/fire_cnn/raw/ak_ca_grd_west_largest2")
var territory22 = ee.FeatureCollection("users/spotter/fire_cnn/raw/ak_ca_grd_west_largest2")

var water = water.filterDate(String(year) + '-01-01', String(year) + '-12-31').max().select('waterClass').clip(geometry)
var water = water.updateMask(water.eq(3))

//modis land cover for now
// var lc = ee.ImageCollection("MODIS/061/MCD12Q1")
//clip lc
// lc = lc.filterDate(String(year) + '-01-01', String(year) + '-12-31').max().clip(table.union())
// lc = lc.select('LC_Type1')

//cropland
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
                               
// Map.addLayer(crop, {min: 0, max: 1})

//read in gabam which is a 30m random forest model
var gabam = ee.ImageCollection("projects/sat-io/open-datasets/GABAM")

gabam = gabam.filterDate(String(year) + '-01-01', String(year) + '-12-31').mean()




//union to a single feature for the table
var aoi = geometry

 


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

// Map.addLayer(mod_fire, {}, 'Mod_fire')
// Map.addLayer(buff_img, {}, 'Buff_img')
var mod_lfdb = ee.ImageCollection([mod_fire.cast({'constant': 'short'}), buff_img.cast({'constant': 'short'})]).max()

// Map.addLayer(mod_lfdb, {}, 'mod_lfdb_orig')
// Map.addLayer(buff_img, {}, 'buff_img')
// Map.addLayer(mod_lfdb, {}, 'mod_lfdb')

// read in viirs
var viirs = ee.Image("users/spotter/fire_cnn/VIIRS/" + String(year))
var viirs = viirs.clip(geometry).select(['b1'], ['constant']).cast({'constant': 'short'})

mod_lfdb = mod_lfdb.cast({'constant': 'short'})

//merge all
// var final = ee.ImageCollection([mod_lfdb, viirs]).max().updateMask(lc.neq(12).and(lc.neq(16))).cast({'constant': 'short'})
// var final = ee.ImageCollection([mod_lfdb, viirs]).max().cast({'constant': 'short'})
// Map.addLayer(mod_lfdb, {}, 'Mod_LFDB')


//large fire databases which is mtbs and nbac
var lfdb = ee.FeatureCollection('users/spotter/fire_cnn/raw/ak_ca_1985')

var lfdb_orig = lfdb.filter(ee.Filter.eq("Year", year))
Map.addLayer(lfdb_orig, {}, 'LFDB Orig')


//all thes are good 893, 889, 891 for tundra
//5147 is tundra kinda
var id = 889
lfdb = lfdb_orig.filter(ee.Filter.eq("ID", id))

var bbox = lfdb.geometry().bounds().buffer(2000, 1)

//just for tundra, otherwise too big
var bbox = lfdb.geometry().bounds().buffer(40000, 1)


Map.addLayer(bbox, {}, 'Bbox')

// Map.addLayer(lfdb)

Map.addLayer(lfdb, {}, 'LFDB')
//make image of fire databases
var lfdb_img = lfdb_orig.reduceToImage({
  properties: ['Year'],
  reducer: ee.Reducer.max()
}).clip(aoi).select(['max'], ['constant']);

//water mask
// Map.addLayer(predictions)
// lfdb_img = lfdb_img.updateMask(water.unmask().not())
// lfdb_img = lfdb_img.updateMask(predictions.gte(0))

mod_lfdb = ee.ImageCollection([mod_lfdb.cast({'constant': 'short'}), lfdb_img.cast({'constant': 'short'})]).max();

var final = ee.Image(ee.Algorithms.If(year <2012, mod_lfdb,  ee.ImageCollection([mod_lfdb, viirs]).max()))


final = final.cast({'constant': 'short'})



// .updateMask(lc.neq(12).and(lc.neq(16)))
// var final = mod_lfdb.cast({'constant': 'short'})


// read in the predictions and mask them with the areas
// var predictions = ee.ImageCollection("projects/gee-serdp-upload/assets/cnn_mapping/ak_ca" + String(year) + "_preds_128_32").max()
// var predictions = ee.Image('users/spotter/fire_cnn/ak_ca_predictions/ak_ca_' + String(year) + '_preds').updateMask(water.unmask().not())
// var predictions = ee.Image("users/spotter/fire_cnn/ak_ca_predictions/ak_ca_" + String(year) + "_preds").updateMask(water.unmask().not())
var predictions = ee.ImageCollection('projects/gee-serdp-upload/assets/cnn_mapping/ak_ca_' + String(year) + '_preds_VI').max()//.updateMask(water.unmask().not())
//dnbr
var dnbr = ee.ImageCollection('users/spotter/fire_cnn/ak_ca_VI/' + String(year)).mean().select('NBR')
var dndvi = ee.ImageCollection('users/spotter/fire_cnn/ak_ca_VI/' + String(year)).mean().select('NDVI')
var dndii= ee.ImageCollection('users/spotter/fire_cnn/ak_ca_VI/' + String(year)).mean().select('NDII')

Map.addLayer(dnbr, {min: 0, max: 1000, palette: dnbr_palette}, 'dNBR')
Map.addLayer(dndvi, {min: 0, max: 1000, palette: dndvi_palette}, 'dNDVI')
Map.addLayer(dndii, {min: 0, max: 1000, palette: dndii_palette}, 'dNDII')

var vis = dnbr.addBands(dndvi).addBands(dndii).clip(bbox).multiply(1000)
// Map.addLayer(predictions, {min: 0, max:1, palette : prediction_palette}, 'Raw Predictions')

Map.addLayer(final, {palette: 'gray'}, 'Constrained Prediction Locations')

Map.addLayer(predictions, {}, 'Raw predictions')




//threshold at 0.5 probability
var thresholded = predictions.select('prediction').updateMask(final).updateMask(predictions.select('prediction').gte(500)).updateMask(crop.neq(1))

//water mask
// thresholded = thresholded.updateMask(water.unmask().not()).multiply(1000).toShort()
thresholded = thresholded.multiply(1000).toShort()

Map.addLayer(thresholded, {palette: 'red'}, 'Final Predictions')

//remove areas from gabam where I didn't have imagery
// gabam = gabam.updateMask(predictions.)


gabam = gabam.updateMask(predictions.gte(0))//.updateMask(lc.neq(12).and(lc.neq(16)))
Map.addLayer(gabam, {palette: 'purple'}, 'GABAM')



var final2 = thresholded.multiply(1000).toShort().addBands(gabam.toShort()).addBands(dnbr.toShort()).addBands(dndvi.toShort()).addBands(dndii.toShort()).clip(bbox).updateMask(final)
Map.addLayer(final2, {}, 'Final')

// //get intersection of us and mtbs
// var us_int = lfdb_img.updateMask(predictions.gte(0.5)).select(['constant'], ['us_int'])
// Map.addLayer(us_int, {palette: 'red'}, 'Us Intersection-Red')

// //get omission of us and mtbs (mtbs has fire we do not)
// var us_om = lfdb_img.updateMask(predictions.gte(0.5).unmask().not()).select(['constant'], ['us_com'])
// Map.addLayer(us_om, {palette: 'purple'}, 'Us Commission-Purple')

// //get comission of us and mtbs  (we have fire but mtbs does not)
// var us_com = thresholded.updateMask(lfdb_img.unmask().not()).select(['prediction'], ['us_om'])
// Map.addLayer(us_com, {palette: 'green'}, 'Us Omission-Green')

// //get intersection of gabam and mtbs
// var gabam_int = lfdb_img.updateMask(gabam).select(['constant'], ['gabam_int'])
// Map.addLayer(gabam_int, {palette: 'Orange'}, 'Gabam Intersection-Orange')

// //get omission of gabam and mtbs
// var gabam_om = lfdb_img.updateMask(gabam.unmask().not()).select(['constant'], ['gabam_com'])
// Map.addLayer(gabam_om, {palette: 'yellow'}, 'Gabam Commission-Yellow')

// //get comission of gabam and mtbs
// var gabam_com = gabam.updateMask(lfdb_img.unmask().not()).select(['b1'], ['gabam_om'])
// Map.addLayer(gabam_com, {palette: 'pink'}, 'Gabam Omission-Pink')




// //combine all layers into one and export for viz
// var for_export = lfdb_img.select(['constant'], ['lfdb']).
//             addBands(final).
//             addBands(us_int).
//             addBands(us_om).
//             addBands(us_com).
//             addBands(gabam_int).
//             addBands(gabam_om).
//             addBands(gabam_com)



// print(for_export)
Export.image.toCloudStorage({
  image: vis,
  description: 'for_map_' + String(year) + '_' + String(id) + '_vis',
  bucket: 'smp-scratch',
  scale: 30,
  crs: 'EPSG:3413',
  // crsTransform: transform,
  maxPixels: 1e13,
  region : bbox,
  fileFormat: 'GeoTIFF'

});

Export.image.toCloudStorage({
  image: final2,
  description: 'for_map_' + String(year) + '_' + String(id),
  bucket: 'smp-scratch',
  scale: 30,
  crs: 'EPSG:3413',
  // crsTransform: transform,
  maxPixels: 1e13,
  region : bbox,
  fileFormat: 'GeoTIFF'

});