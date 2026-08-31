// Task 6A basic independent covariates: 3 products x 4 city-code batches.
// Run all 12 exports. Polycentricity and periurban farm are handled separately.
var RUN_ID = 'task6_covariates_basic_296_20260812_r1';
var ASSET_ID = 'projects/YOUR_EE_PROJECT/assets/GCTB/task6_reference_admin2021';
var BATCHES = [
  {id: 'b1_11_29', lower: 'CN-110000', upper: 'CN-300000'},
  {id: 'b2_31_39', lower: 'CN-310000', upper: 'CN-400000'},
  {id: 'b3_41_49', lower: 'CN-410000', upper: 'CN-500000'},
  {id: 'b4_50_65', lower: 'CN-500000', upper: 'CN-660000'}
];

function polygonalGeometry(rawGeometry) {
  var rawType = rawGeometry.type();
  var children = ee.List(ee.Algorithms.If(
    rawType.equals('GeometryCollection'), rawGeometry.geometries(), [rawGeometry]
  ));
  var parts = ee.FeatureCollection(children.map(function(child) {
    var part = ee.Geometry(child);
    return ee.Feature(part, {part_type: part.type()});
  }));
  var polygonParts = parts.filter(ee.Filter.inList(
    'part_type', ['Polygon', 'MultiPolygon']
  ));
  return ee.Geometry(ee.Algorithms.If(
    rawType.equals('GeometryCollection'), polygonParts.geometry(1), rawGeometry
  )).dissolve({maxError: 1});
}

var boundaries = ee.FeatureCollection(ASSET_ID).map(function(feature) {
  return ee.Feature(polygonalGeometry(feature.geometry()), {
    city_id: feature.get('city_id'),
    city_name: feature.get('city_name'),
    src_code: feature.get('src_code'),
    src_year: feature.get('src_year')
  });
});
print('Normalized reference city count (expected 296)', boundaries.size());

var worldPopCollection = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filter(ee.Filter.eq('country', 'CHN'))
  .filter(ee.Filter.eq('year', 2020));
print('WorldPop 2020 China image count (expected 1)', worldPopCollection.size());
var worldPop = worldPopCollection.first().select('population').rename('population_2020');

var yearlyWater = ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
  .filterDate('2021-01-01', '2022-01-01').first().select('waterClass');
var waterValid = yearlyWater.gt(0);
var waterAreaImage = ee.Image.cat([
  ee.Image.pixelArea().updateMask(yearlyWater.eq(3)).rename('permanent_water_area_m2'),
  // This is classified-support area under the product mask, not the density denominator.
  // The frozen density denominator is full 2021 reference-city area.
  ee.Image.pixelArea().updateMask(waterValid).rename('water_valid_area_m2')
]);

var dem = ee.Image('MERIT/DEM/v1_0_3').select('dem');
var neighborhood = dem.neighborhoodToBands(ee.Kernel.square({radius: 1, units: 'pixels'}));
var tri = neighborhood.subtract(dem).pow(2).reduce(ee.Reducer.sum()).sqrt()
  .rename('terrain_ruggedness_riley_m');

function exportReduced(image, reducer, scale, productId, selectors) {
  BATCHES.forEach(function(batch) {
    var cities = boundaries
      .filter(ee.Filter.gte('city_id', batch.lower))
      .filter(ee.Filter.lt('city_id', batch.upper));
    print(productId + ' / ' + batch.id + ' city count', cities.size());
    var output = image.reduceRegions({
      collection: cities,
      reducer: reducer,
      scale: scale,
      tileScale: 16,
      maxPixelsPerRegion: 10000000000
    }).map(function(feature) {
      return ee.Feature(null, feature.toDictionary()).set({
        run_id: RUN_ID,
        batch_id: batch.id,
        covariate_product_id: productId
      });
    });
    var name = 'R1_task6_' + productId + '_' + batch.id;
    Export.table.toDrive({
      collection: output,
      description: name,
      folder: 'GCTB_rebuild',
      fileNamePrefix: name,
      fileFormat: 'CSV',
      selectors: ['run_id', 'batch_id', 'covariate_product_id',
        'city_id', 'city_name', 'src_code', 'src_year'].concat(selectors)
    });
  });
}

exportReduced(worldPop, ee.Reducer.sum().setOutputs(['population_2020']),
  100, 'worldpop2020_population', [
  'population_2020'
]);
exportReduced(waterAreaImage, ee.Reducer.sum(), 30, 'jrcgsw2021_permanent_water', [
  'permanent_water_area_m2', 'water_valid_area_m2'
]);
exportReduced(tri, ee.Reducer.mean().setOutputs(['terrain_ruggedness_riley_m']),
  90, 'meritdem_terrain_ruggedness', [
  'terrain_ruggedness_riley_m'
]);
