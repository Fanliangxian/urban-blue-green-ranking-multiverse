// R2 Figure 1 land-cover example for the geometry-selected city (Shangluo).
// Exports two co-registered 10 m rasters and one audit CSV to Google Drive.
// Dynamic World remains a continuous annual-mean blue-green probability
// contribution; it is deliberately not converted to a categorical winner map.

var RUN_ID = 'r2_figure1_landcover_example_20260820';
var CITY_ID = 'CN-611000';
var CITY_NAME = 'Shangluo';
var BOUNDARY_ASSET =
    'projects/YOUR_EE_PROJECT/assets/GCTB/R1_measurement_boundaries_296';
var EXPORT_FOLDER = 'GCTB_rebuild';
var EXPORT_CRS = 'EPSG:32649';  // UTM zone 49N, appropriate for Shangluo.
var EXPORT_SCALE_M = 10;
var BUFFER_M = 5000;

var raw = ee.FeatureCollection(BOUNDARY_ASSET)
    .filter(ee.Filter.eq('city_id', CITY_ID))
    .filter(ee.Filter.eq('geom_var', 'raw_product_geometry'));

print('Representative city', CITY_ID, CITY_NAME);
print('Raw boundary count (expected 4)', raw.size());
print('Raw boundary scenarios', raw.aggregate_array('bnd_scn').sort());

// One deterministic rectangle shared by both raster exports. The buffer prevents
// the most expansive boundary from touching the frame.
var exportProjection = ee.Projection(EXPORT_CRS).atScale(EXPORT_SCALE_M);
var exportRegion = raw.geometry(1)
    .buffer(BUFFER_M, 1)
    .bounds(1, exportProjection);

var worldCover = ee.ImageCollection('ESA/WorldCover/v200')
    .first()
    .select('Map')
    .rename('worldcover_class_2021')
    .toUint8();

var dynamicWorld = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterDate('2021-01-01', '2022-01-01')
    .filterBounds(exportRegion);

var blueProbability = dynamicWorld.select('water').mean();
var greenProbability = dynamicWorld
    .select(['trees', 'grass', 'flooded_vegetation', 'shrub_and_scrub'])
    .mean()
    .reduce(ee.Reducer.sum());
var blueGreenProbability = blueProbability.add(greenProbability)
    .rename('annual_mean_blue_green_probability_2021')
    .toFloat();

print('Dynamic World 2021 scene count (expected > 0)', dynamicWorld.size());
print('Export projection', exportProjection);
print('Export region', exportRegion);

var manifest = ee.FeatureCollection([ee.Feature(null, {
  run_id: RUN_ID,
  city_id: CITY_ID,
  city_name_en: CITY_NAME,
  selection_rule: 'D-054 geometry-only representative-city score',
  boundary_asset: BOUNDARY_ASSET,
  boundary_geometry_variant: 'raw_product_geometry',
  boundary_count_expected: 4,
  extent_rule: 'union of four raw boundaries buffered 5000 m then bounded',
  export_crs: EXPORT_CRS,
  export_scale_m: EXPORT_SCALE_M,
  worldcover_dataset_id: 'ESA/WorldCover/v200',
  worldcover_band: 'Map',
  worldcover_semantics: 'categorical native class label',
  dynamic_world_dataset_id: 'GOOGLE/DYNAMICWORLD/V1',
  dynamic_world_start: '2021-01-01',
  dynamic_world_end_exclusive: '2022-01-01',
  dynamic_world_semantics:
      'annual mean of water + trees + grass + flooded_vegetation + shrub_and_scrub probabilities',
  probability_value_min: 0,
  probability_value_max: 1,
  export_region_area_m2: exportRegion.area(1)
})]);

Export.image.toDrive({
  image: worldCover,
  description: 'R2_figure1_worldcover2021_shangluo',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'R2_figure1_worldcover2021_shangluo',
  region: exportRegion,
  crs: EXPORT_CRS,
  scale: EXPORT_SCALE_M,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: 0}
});

Export.image.toDrive({
  image: blueGreenProbability,
  description: 'R2_figure1_dynamicworld2021_bluegreen_probability_shangluo',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'R2_figure1_dynamicworld2021_bluegreen_probability_shangluo',
  region: exportRegion,
  crs: EXPORT_CRS,
  scale: EXPORT_SCALE_M,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true, noData: -9999}
});

Export.table.toDrive({
  collection: manifest,
  description: 'R2_figure1_landcover_example_manifest',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'R2_figure1_landcover_example_manifest',
  fileFormat: 'CSV',
  selectors: [
    'run_id', 'city_id', 'city_name_en', 'selection_rule', 'boundary_asset',
    'boundary_geometry_variant', 'boundary_count_expected', 'extent_rule',
    'export_crs', 'export_scale_m', 'worldcover_dataset_id', 'worldcover_band',
    'worldcover_semantics', 'dynamic_world_dataset_id', 'dynamic_world_start',
    'dynamic_world_end_exclusive', 'dynamic_world_semantics',
    'probability_value_min', 'probability_value_max', 'export_region_area_m2'
  ]
});
