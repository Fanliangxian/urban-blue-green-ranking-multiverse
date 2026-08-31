// Dynamic World 2021 probability-area national run r3.
// Four boundary scenarios x four deterministic city-code batches = 16 exports.
var RUN_ID = 'dynamicworld2021_probability_296_20260806_r3';
var BOUNDARY_ASSET_ID = 'projects/YOUR_EE_PROJECT/assets/GCTB/R1_measurement_boundaries_296';
var SCENARIOS = [
  {id: 'gctb_core_2021', expected: 287},
  {id: 'gctb_system_2021', expected: 287},
  {id: 'gub_2018', expected: 294},
  {id: 'ghs_uc_2020', expected: 290}
];
// Half-open lexicographic ranges over the fixed CN-###### city identifiers.
var BATCHES = [
  {id: 'b1_11_29', lower: 'CN-110000', upper: 'CN-300000'},
  {id: 'b2_31_39', lower: 'CN-310000', upper: 'CN-400000'},
  {id: 'b3_41_49', lower: 'CN-410000', upper: 'CN-500000'},
  {id: 'b4_50_65', lower: 'CN-500000', upper: 'CN-660000'}
];
var PROBABILITY_BANDS = [
  'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
  'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
];
var SELECTORS = [
  'run_id', 'batch_id', 'uid', 'city_id', 'bnd_scn', 'geom_var', 'lic_ok', 'formal_ok',
  'landcover_dataset_id', 'estimator_id', 'measurement_status',
  'boundary_area_m2', 'valid_area_m2', 'raw_valid_fraction', 'valid_fraction',
  'boundary_minus_valid_area_m2', 'unclassified_area_m2',
  'raster_area_excess_m2', 'raster_area_excess_fraction',
  'Blue_m2', 'Green_m2', 'Grey_m2', 'Farm_m2', 'Other_Uncertain_m2',
  'class_sum_m2', 'mass_balance_error_m2', 'mass_balance_relative_error',
  'probability_closure_error_area_m2', 'probability_closure_relative_error',
  'Blue_share_full_boundary', 'Green_share_full_boundary',
  'Grey_share_full_boundary', 'Farm_share_full_boundary',
  'Other_Uncertain_share_full_boundary',
  'Blue_share_valid_area', 'Green_share_valid_area', 'Grey_share_valid_area',
  'Farm_share_valid_area', 'Other_Uncertain_share_valid_area'
];

var allBoundaries = ee.FeatureCollection(BOUNDARY_ASSET_ID)
  .filter(ee.Filter.eq('geom_var', 'raw_product_geometry'));
var chinaRectangle = ee.Geometry.Rectangle([73, 18, 135, 54], null, false);
var dynamicWorld = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate('2021-01-01', '2022-01-01')
  .filterBounds(chinaRectangle)
  .select(PROBABILITY_BANDS);

// Full WKT avoids dependence on runtime CRS authority-code resolution.
var MOLLWEIDE_WKT =
  'PROJCS["World_Mollweide",' +
  'GEOGCS["GCS_WGS_1984",' +
  'DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],' +
  'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],' +
  'PROJECTION["Mollweide"],' +
  'PARAMETER["False_Easting",0],' +
  'PARAMETER["False_Northing",0],' +
  'PARAMETER["Central_Meridian",0],' +
  'UNIT["Meter",1]]';
var analysisProjection = ee.Projection(MOLLWEIDE_WKT).atScale(10);
var probabilities = dynamicWorld.mean().setDefaultProjection(analysisProjection);
var valid = probabilities.select('water').mask().gt(0);
var pixelArea = ee.Image.pixelArea();

var blueP = probabilities.select('water').rename('Blue_p');
var greenP = probabilities.select([
  'trees', 'grass', 'flooded_vegetation', 'shrub_and_scrub'
]).reduce(ee.Reducer.sum()).rename('Green_p');
var greyP = probabilities.select('built').rename('Grey_p');
var farmP = probabilities.select('crops').rename('Farm_p');
var otherP = probabilities.select(['bare', 'snow_and_ice'])
  .reduce(ee.Reducer.sum()).rename('Other_Uncertain_p');
var targetProbabilities = ee.Image.cat([blueP, greenP, greyP, farmP, otherP]);
var probabilitySum = targetProbabilities.reduce(ee.Reducer.sum());
var areaImage = ee.Image.cat([
  pixelArea.updateMask(valid).rename('valid_area_m2'),
  blueP.multiply(pixelArea).updateMask(valid).rename('Blue_m2'),
  greenP.multiply(pixelArea).updateMask(valid).rename('Green_m2'),
  greyP.multiply(pixelArea).updateMask(valid).rename('Grey_m2'),
  farmP.multiply(pixelArea).updateMask(valid).rename('Farm_m2'),
  otherP.multiply(pixelArea).updateMask(valid).rename('Other_Uncertain_m2'),
  probabilitySum.subtract(1).abs().multiply(pixelArea).updateMask(valid)
    .rename('probability_closure_error_area_m2')
]).setDefaultProjection(analysisProjection);

function numberOrZero(feature, propertyName) {
  return ee.Number(ee.Dictionary(feature.toDictionary()).get(propertyName, 0));
}

function finalize(feature, batchId) {
  var boundaryArea = ee.Number(feature.get('bnd_area'));
  var validArea = numberOrZero(feature, 'valid_area_m2');
  var blue = numberOrZero(feature, 'Blue_m2');
  var green = numberOrZero(feature, 'Green_m2');
  var grey = numberOrZero(feature, 'Grey_m2');
  var farm = numberOrZero(feature, 'Farm_m2');
  var other = numberOrZero(feature, 'Other_Uncertain_m2');
  var classSum = blue.add(green).add(grey).add(farm).add(other);
  var rawValidFraction = ee.Number(ee.Algorithms.If(
    boundaryArea.gt(0), validArea.divide(boundaryArea), 0
  ));
  var validFraction = rawValidFraction.max(0).min(1);
  var boundaryMinusValid = boundaryArea.subtract(validArea);
  var rasterExcess = validArea.subtract(boundaryArea).max(0);
  var validDenominator = ee.Number(ee.Algorithms.If(validArea.gt(0), validArea, 1));
  var status = ee.String(ee.Algorithms.If(
    validArea.eq(0), 'no_valid_pixels',
    ee.Algorithms.If(
      validFraction.lt(0.95), 'low_valid_coverage',
      ee.Algorithms.If(validFraction.lt(0.999), 'partial_valid_coverage', 'complete_valid_coverage')
    )
  ));
  return ee.Feature(null, feature.toDictionary()).set({
    run_id: RUN_ID,
    batch_id: batchId,
    landcover_dataset_id: 'dynamic_world_2021',
    estimator_id: 'annual_mean_probability_area',
    measurement_status: status,
    boundary_area_m2: boundaryArea,
    raw_valid_fraction: rawValidFraction,
    valid_fraction: validFraction,
    boundary_minus_valid_area_m2: boundaryMinusValid,
    unclassified_area_m2: boundaryMinusValid.max(0),
    raster_area_excess_m2: rasterExcess,
    raster_area_excess_fraction: ee.Number(ee.Algorithms.If(
      boundaryArea.gt(0), rasterExcess.divide(boundaryArea), 0
    )),
    class_sum_m2: classSum,
    mass_balance_error_m2: classSum.subtract(validArea),
    mass_balance_relative_error: classSum.subtract(validArea).abs().divide(validDenominator),
    probability_closure_relative_error: numberOrZero(
      feature, 'probability_closure_error_area_m2'
    ).divide(validDenominator),
    Blue_share_full_boundary: blue.divide(boundaryArea),
    Green_share_full_boundary: green.divide(boundaryArea),
    Grey_share_full_boundary: grey.divide(boundaryArea),
    Farm_share_full_boundary: farm.divide(boundaryArea),
    Other_Uncertain_share_full_boundary: other.divide(boundaryArea),
    Blue_share_valid_area: blue.divide(validDenominator),
    Green_share_valid_area: green.divide(validDenominator),
    Grey_share_valid_area: grey.divide(validDenominator),
    Farm_share_valid_area: farm.divide(validDenominator),
    Other_Uncertain_share_valid_area: other.divide(validDenominator)
  });
}

print('National raw boundary count (expected 1158)', allBoundaries.size());
// The registered population is 296 cities, but Changdu and Nyingchi have no
// positive-overlap geometry in any of the four frozen boundary scenarios.
print('National cities represented by at least one raw boundary (expected 294)',
  allBoundaries.aggregate_count_distinct('city_id'));
print('Frozen analysis projection', analysisProjection);

SCENARIOS.forEach(function(scenario) {
  var scenarioBoundaries = allBoundaries.filter(ee.Filter.eq('bnd_scn', scenario.id));
  print(scenario.id + ' total boundary count (expected ' + scenario.expected + ')',
    scenarioBoundaries.size());
  BATCHES.forEach(function(batch) {
    var boundaries = scenarioBoundaries
      .filter(ee.Filter.gte('city_id', batch.lower))
      .filter(ee.Filter.lt('city_id', batch.upper));
    print(scenario.id + ' / ' + batch.id + ' count', boundaries.size());
    var output = areaImage.reduceRegions({
      collection: boundaries,
      reducer: ee.Reducer.sum(),
      scale: 10,
      crs: analysisProjection,
      tileScale: 16,
      maxPixelsPerRegion: 10000000000
    }).map(function(feature) {
      return finalize(feature, batch.id);
    });
    var exportName = 'R1_dw2021_prob_296_r3_' + scenario.id + '_' + batch.id;
    Export.table.toDrive({
      collection: output,
      description: exportName,
      folder: 'GCTB_rebuild',
      fileNamePrefix: exportName,
      fileFormat: 'CSV',
      selectors: SELECTORS
    });
  });
});
