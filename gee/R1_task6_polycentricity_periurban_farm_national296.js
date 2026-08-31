// Task 6A national extraction for the frozen 296-city registry.
// Batch size 6 was frozen after exact agreement among the r4 one-city baseline
// and independent 2-, 4-, and 6-city capacity exports.

var RUN_ID = 'task6_polycentricity_periurban_farm_national296_20260818_r1';
var ASSET_ID = 'projects/YOUR_EE_PROJECT/assets/GCTB/task6_reference_admin2021';
var EXPECTED_CITY_COUNT = 296;
var BATCH_SIZE = 6;
var GRID_M = 1000;
var URBAN_DENSITY_THRESHOLD = 1500;
var MIN_CENTER_POPULATION = 50000;
var PERIURBAN_BUFFER_M = 5000;

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
var gridProjection = ee.Projection(MOLLWEIDE_WKT).atScale(GRID_M);

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

var reference = ee.FeatureCollection(ASSET_ID).map(function(feature) {
  return ee.Feature(polygonalGeometry(feature.geometry()), feature.toDictionary());
});
var worldPop = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filter(ee.Filter.eq('country', 'CHN'))
  .filter(ee.Filter.eq('year', 2020)).first().select('population');
var population1km = worldPop.divide(ee.Image.pixelArea())
  .rename('people_per_m2').reduceResolution({
    reducer: ee.Reducer.mean(), maxPixels: 1024
  }).reproject({crs: gridProjection})
  .multiply(ee.Image.pixelArea().reproject({crs: gridProjection}))
  .rename('population_1km');
var cropFraction = ee.ImageCollection(
  'COPERNICUS/Landcover/100m/Proba-V-C3/Global'
).filterDate('2019-01-01', '2020-01-01').first()
  .select('crops-coverfraction').divide(100).rename('crop_fraction');

function numberOrZero(dictionary, key) {
  return ee.Number(ee.Dictionary(dictionary).get(key, 0));
}

function measureCity(feature) {
  feature = ee.Feature(feature);
  var geometry = feature.geometry();
  var pop = population1km.clip(geometry);
  var urban = pop.gte(URBAN_DENSITY_THRESHOLD).selfMask().rename('urban');
  var componentGeometries = urban.toInt()
    .addBands(pop.rename('component_population')).reduceToVectors({
      reducer: ee.Reducer.sum().setOutputs(['center_population']),
      geometry: geometry, crs: gridProjection, scale: GRID_M,
      geometryType: 'polygon', eightConnected: true,
      labelProperty: 'urban_label', maxPixels: 1e9, tileScale: 8
    });
  var components = worldPop.rename('center_population').reduceRegions({
    collection: componentGeometries,
    reducer: ee.Reducer.sum().setOutputs(['center_population']),
    scale: 100, tileScale: 16
  });
  var retainedCenters = components.filter(
    ee.Filter.gte('center_population', MIN_CENTER_POPULATION)
  );
  var centerCount = retainedCenters.size();
  var totalRetained = ee.Number(ee.Algorithms.If(
    centerCount.gt(0), retainedCenters.aggregate_sum('center_population'), 0
  ));
  var centerHHI = ee.Number(ee.Algorithms.If(centerCount.gt(0),
    retainedCenters.map(function(center) {
      var share = ee.Number(center.get('center_population')).divide(totalRetained);
      return center.set('squared_share', share.pow(2));
    }).aggregate_sum('squared_share'), 0));
  var polycentricity = ee.Number(ee.Algorithms.If(
    centerCount.gte(2), ee.Number(1).subtract(centerHHI), 0
  ));
  var retainedBinary = ee.Image(0).byte().paint(retainedCenters, 1)
    .reproject({crs: gridProjection}).clip(geometry).gt(0);
  var periurbanRing = retainedBinary.focalMax({
    radius: PERIURBAN_BUFFER_M, units: 'meters', kernelType: 'circle'
  }).and(retainedBinary.not()).clip(geometry).selfMask();
  var periurbanAreas = ee.Image.cat([
    ee.Image.pixelArea().updateMask(periurbanRing).rename('ring_area_m2'),
    cropFraction.multiply(ee.Image.pixelArea()).updateMask(periurbanRing)
      .rename('crop_area_equivalent_m2')
  ]).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: geometry, scale: 100,
    maxPixels: 1e10, tileScale: 16
  });
  var ringArea = numberOrZero(periurbanAreas, 'ring_area_m2');
  var cropArea = numberOrZero(periurbanAreas, 'crop_area_equivalent_m2');
  return ee.Feature(null, {
    run_id: RUN_ID, city_id: feature.get('city_id'),
    city_name: feature.get('city_name'), urban_grid_m: GRID_M,
    urban_density_threshold_people_km2: URBAN_DENSITY_THRESHOLD,
    minimum_center_population: MIN_CENTER_POPULATION,
    center_count: centerCount, retained_center_population: totalRetained,
    center_population_hhi: centerHHI,
    polycentricity_index: polycentricity,
    periurban_buffer_m: PERIURBAN_BUFFER_M,
    periurban_ring_area_m2: ringArea,
    periurban_crop_area_equivalent_m2: cropArea,
    periurban_farm_fraction: ee.Number(ee.Algorithms.If(
      ringArea.gt(0), cropArea.divide(ringArea), 0
    ))
  });
}

var SELECTORS = [
  'run_id', 'city_id', 'city_name', 'urban_grid_m',
  'urban_density_threshold_people_km2', 'minimum_center_population',
  'center_count', 'retained_center_population', 'center_population_hhi',
  'polycentricity_index', 'periurban_buffer_m', 'periurban_ring_area_m2',
  'periurban_crop_area_equivalent_m2', 'periurban_farm_fraction'
];

function exportBatch(ids, batchNumber) {
  var cities = reference.filter(ee.Filter.inList('city_id', ids));
  var output = cities.map(measureCity);
  var batchLabel = ('000' + batchNumber).slice(-3);
  var name = 'R1_task6_polycentricity_periurban_farm_national_r1_batch_' +
    batchLabel;
  Export.table.toDrive({
    collection: output, description: name, folder: 'GCTB_rebuild',
    fileNamePrefix: name, fileFormat: 'CSV', selectors: SELECTORS
  });
}

print('National reference city count (expected 296)', reference.size());
print('Frozen batch size', BATCH_SIZE);
print('Expected task count', Math.ceil(EXPECTED_CITY_COUNT / BATCH_SIZE));
print('Run guidance',
  'Start tasks sequentially or with only a small number active; all 50 tasks must finish.');

// Export declarations require client-side batches. The asset supplies the IDs;
// sorting freezes deterministic membership without duplicating the registry in
// this script. The callback should create exactly 50 Tasks after evaluation.
reference.aggregate_array('city_id').sort().evaluate(function(cityIds, error) {
  if (error) {
    print('ERROR: could not retrieve national city IDs', error);
    return;
  }
  if (cityIds.length !== EXPECTED_CITY_COUNT) {
    print('ERROR: refusing export because city count is not 296', cityIds.length);
    return;
  }
  var taskCount = Math.ceil(cityIds.length / BATCH_SIZE);
  print('Client-side city count', cityIds.length);
  print('Tasks created (expected 50)', taskCount);
  for (var start = 0; start < cityIds.length; start += BATCH_SIZE) {
    var ids = cityIds.slice(start, start + BATCH_SIZE);
    exportBatch(ids, Math.floor(start / BATCH_SIZE) + 1);
  }
});
