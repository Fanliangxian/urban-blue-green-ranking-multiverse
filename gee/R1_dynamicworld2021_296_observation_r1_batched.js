// Dynamic World 2021 national observation-support audit.
// Four boundary scenarios x four deterministic city-code batches = 16 exports.
var RUN_ID = 'dynamicworld2021_observation_296_20260807_r1';
var BOUNDARY_ASSET_ID = 'projects/YOUR_EE_PROJECT/assets/GCTB/R1_measurement_boundaries_296';
var SCENARIOS = [
  {id: 'gctb_core_2021', expected: 287},
  {id: 'gctb_system_2021', expected: 287},
  {id: 'gub_2018', expected: 294},
  {id: 'ghs_uc_2020', expected: 290}
];
var BATCHES = [
  {id: 'b1_11_29', lower: 'CN-110000', upper: 'CN-300000'},
  {id: 'b2_31_39', lower: 'CN-310000', upper: 'CN-400000'},
  {id: 'b3_41_49', lower: 'CN-410000', upper: 'CN-500000'},
  {id: 'b4_50_65', lower: 'CN-500000', upper: 'CN-660000'}
];
var SELECTORS = [
  'run_id', 'batch_id', 'uid', 'city_id', 'bnd_scn', 'boundary_area_m2',
  'obs_0_area_m2', 'obs_1_4_area_m2', 'obs_5_9_area_m2',
  'obs_10_19_area_m2', 'obs_ge20_area_m2', 'observed_area_m2',
  'obs_0_fraction', 'obs_1_4_fraction', 'obs_5_9_fraction',
  'obs_10_19_fraction', 'obs_ge20_fraction',
  'mean_observation_count_on_observed_area', 'observation_bin_closure_error_m2',
  'observation_bin_closure_relative_error'
];

var allBoundaries = ee.FeatureCollection(BOUNDARY_ASSET_ID)
  .filter(ee.Filter.eq('geom_var', 'raw_product_geometry'));
var chinaRectangle = ee.Geometry.Rectangle([73, 18, 135, 54], null, false);
var dynamicWorld = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate('2021-01-01', '2022-01-01')
  .filterBounds(chinaRectangle)
  .select('water');
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
var count = dynamicWorld.count().unmask(0).setDefaultProjection(analysisProjection);
var pixelArea = ee.Image.pixelArea();
var observed = count.gt(0);
var diagnosticImage = ee.Image.cat([
  pixelArea.updateMask(count.eq(0)).rename('obs_0_area_m2'),
  pixelArea.updateMask(count.gte(1).and(count.lte(4))).rename('obs_1_4_area_m2'),
  pixelArea.updateMask(count.gte(5).and(count.lte(9))).rename('obs_5_9_area_m2'),
  pixelArea.updateMask(count.gte(10).and(count.lte(19))).rename('obs_10_19_area_m2'),
  pixelArea.updateMask(count.gte(20)).rename('obs_ge20_area_m2'),
  pixelArea.updateMask(observed).rename('observed_area_m2'),
  count.multiply(pixelArea).updateMask(observed).rename('observation_count_x_area')
]).setDefaultProjection(analysisProjection);

function numberOrZero(feature, propertyName) {
  return ee.Number(ee.Dictionary(feature.toDictionary()).get(propertyName, 0));
}

function finalize(feature, batchId) {
  var boundaryArea = ee.Number(feature.get('bnd_area'));
  var observedArea = numberOrZero(feature, 'observed_area_m2');
  var zero = numberOrZero(feature, 'obs_0_area_m2');
  var oneToFour = numberOrZero(feature, 'obs_1_4_area_m2');
  var fiveToNine = numberOrZero(feature, 'obs_5_9_area_m2');
  var tenToNineteen = numberOrZero(feature, 'obs_10_19_area_m2');
  var twentyPlus = numberOrZero(feature, 'obs_ge20_area_m2');
  var binSum = zero.add(oneToFour).add(fiveToNine).add(tenToNineteen).add(twentyPlus);
  return ee.Feature(null, feature.toDictionary()).set({
    run_id: RUN_ID,
    batch_id: batchId,
    boundary_area_m2: boundaryArea,
    obs_0_fraction: zero.divide(boundaryArea),
    obs_1_4_fraction: oneToFour.divide(boundaryArea),
    obs_5_9_fraction: fiveToNine.divide(boundaryArea),
    obs_10_19_fraction: tenToNineteen.divide(boundaryArea),
    obs_ge20_fraction: twentyPlus.divide(boundaryArea),
    mean_observation_count_on_observed_area: ee.Number(ee.Algorithms.If(
      observedArea.gt(0),
      numberOrZero(feature, 'observation_count_x_area').divide(observedArea),
      0
    )),
    observation_bin_closure_error_m2: binSum.subtract(boundaryArea),
    observation_bin_closure_relative_error: binSum.subtract(boundaryArea)
      .abs().divide(boundaryArea)
  });
}

print('National raw boundary count (expected 1158)', allBoundaries.size());
print('Cities represented by at least one raw boundary (expected 294)',
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
    var output = diagnosticImage.reduceRegions({
      collection: boundaries,
      reducer: ee.Reducer.sum(),
      scale: 10,
      crs: analysisProjection,
      tileScale: 16,
      maxPixelsPerRegion: 10000000000
    }).map(function(feature) {
      return finalize(feature, batch.id);
    });
    var exportName = 'R1_dw2021_obs_296_r1_' + scenario.id + '_' + batch.id;
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
