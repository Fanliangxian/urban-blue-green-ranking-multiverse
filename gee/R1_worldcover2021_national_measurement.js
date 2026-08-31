// WorldCover 2021 national measurement: four separate exports for robust execution.
var RUN_ID = 'worldcover2021_296_20260806_r1';
var BOUNDARY_ASSET_ID = 'projects/YOUR_EE_PROJECT/assets/GCTB/R1_measurement_boundaries_296';
var SCENARIOS = [
  {id: 'gctb_core_2021', expected: 287},
  {id: 'gctb_system_2021', expected: 287},
  {id: 'gub_2018', expected: 294},
  {id: 'ghs_uc_2020', expected: 290}
];
var SELECTORS = [
  'run_id', 'uid', 'city_id', 'bnd_scn', 'geom_var', 'lic_ok', 'formal_ok',
  'landcover_dataset_id', 'estimator_id', 'measurement_status',
  'boundary_area_m2', 'valid_area_m2', 'raw_valid_fraction', 'valid_fraction',
  'boundary_minus_valid_area_m2', 'unclassified_area_m2',
  'raster_area_excess_m2', 'raster_area_excess_fraction',
  'Blue_m2', 'Green_m2', 'Grey_m2', 'Farm_m2', 'Other_Uncertain_m2',
  'unmapped_valid_m2', 'class_sum_m2', 'mass_balance_error_m2',
  'mass_balance_relative_error',
  'Blue_share_full_boundary', 'Green_share_full_boundary',
  'Grey_share_full_boundary', 'Farm_share_full_boundary',
  'Other_Uncertain_share_full_boundary',
  'Blue_share_valid_area', 'Green_share_valid_area', 'Grey_share_valid_area',
  'Farm_share_valid_area', 'Other_Uncertain_share_valid_area'
];

var allRawBoundaries = ee.FeatureCollection(BOUNDARY_ASSET_ID)
  .filter(ee.Filter.eq('geom_var', 'raw_product_geometry'));
var worldCover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var valid = worldCover.mask().gt(0);
// Primary crosswalk: open water=Blue; vegetated wetlands=Green.
var target = worldCover.remap(
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
  [ 2,  2,  2,  4,  3,  5,  5,  1,  2,  2,   5],
  0
).rename('target');
var pixelArea = ee.Image.pixelArea();
var areaImage = ee.Image.cat([
  pixelArea.updateMask(valid).rename('valid_area_m2'),
  pixelArea.updateMask(valid.and(target.eq(1))).rename('Blue_m2'),
  pixelArea.updateMask(valid.and(target.eq(2))).rename('Green_m2'),
  pixelArea.updateMask(valid.and(target.eq(3))).rename('Grey_m2'),
  pixelArea.updateMask(valid.and(target.eq(4))).rename('Farm_m2'),
  pixelArea.updateMask(valid.and(target.eq(5))).rename('Other_Uncertain_m2'),
  pixelArea.updateMask(valid.and(target.eq(0))).rename('unmapped_valid_m2')
]);

function numberOrZero(feature, propertyName) {
  return ee.Number(ee.Dictionary(feature.toDictionary()).get(propertyName, 0));
}

function finalize(feature) {
  var boundaryArea = ee.Number(feature.get('bnd_area'));
  var validArea = numberOrZero(feature, 'valid_area_m2');
  var blue = numberOrZero(feature, 'Blue_m2');
  var green = numberOrZero(feature, 'Green_m2');
  var grey = numberOrZero(feature, 'Grey_m2');
  var farm = numberOrZero(feature, 'Farm_m2');
  var other = numberOrZero(feature, 'Other_Uncertain_m2');
  var unmapped = numberOrZero(feature, 'unmapped_valid_m2');
  var classSum = blue.add(green).add(grey).add(farm).add(other);
  var rawValidFraction = ee.Number(ee.Algorithms.If(
    boundaryArea.gt(0), validArea.divide(boundaryArea), 0
  ));
  var validFraction = rawValidFraction.max(0).min(1);
  var boundaryMinusValid = boundaryArea.subtract(validArea);
  var unclassifiedArea = boundaryMinusValid.max(0);
  var rasterExcess = validArea.subtract(boundaryArea).max(0);
  var rasterExcessFraction = ee.Number(ee.Algorithms.If(
    boundaryArea.gt(0), rasterExcess.divide(boundaryArea), 0
  ));
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
    landcover_dataset_id: 'esa_worldcover_2021',
    estimator_id: 'categorical_native_pixel_area',
    measurement_status: status,
    boundary_area_m2: boundaryArea,
    raw_valid_fraction: rawValidFraction,
    valid_fraction: validFraction,
    boundary_minus_valid_area_m2: boundaryMinusValid,
    unclassified_area_m2: unclassifiedArea,
    raster_area_excess_m2: rasterExcess,
    raster_area_excess_fraction: rasterExcessFraction,
    class_sum_m2: classSum,
    mass_balance_error_m2: classSum.add(unmapped).subtract(validArea),
    mass_balance_relative_error: classSum.add(unmapped).subtract(validArea)
      .abs().divide(validDenominator),
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

print('All raw boundary rows (expected 1158)', allRawBoundaries.size());
print('WorldCover projection', worldCover.projection());

SCENARIOS.forEach(function(spec) {
  var boundaries = allRawBoundaries.filter(ee.Filter.eq('bnd_scn', spec.id));
  print(spec.id + ' boundary count (expected ' + spec.expected + ')', boundaries.size());
  var output = areaImage.reduceRegions({
    collection: boundaries,
    reducer: ee.Reducer.sum(),
    scale: 10,
    crs: worldCover.projection(),
    tileScale: 4,
    maxPixelsPerRegion: 10000000000
  }).map(finalize);
  print(spec.id + ' status histogram', output.aggregate_histogram('measurement_status'));
  print(spec.id + ' maximum unmapped area', output.aggregate_max('unmapped_valid_m2'));
  print(spec.id + ' maximum mass-balance relative error',
    output.aggregate_max('mass_balance_relative_error'));
  print(spec.id + ' maximum raster-area excess fraction',
    output.aggregate_max('raster_area_excess_fraction'));
  Export.table.toDrive({
    collection: output,
    description: 'R1_worldcover2021_296_' + spec.id,
    folder: 'GCTB_rebuild',
    fileNamePrefix: 'R1_worldcover2021_296_' + spec.id,
    fileFormat: 'CSV',
    selectors: SELECTORS
  });
});
