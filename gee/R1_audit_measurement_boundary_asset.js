// Audit the canonical measurement-boundary table after manual GEE upload.
// Replace the sentinel only with the actual uploaded table asset ID.
var BOUNDARY_ASSET_ID = 'PENDING_USER_UPLOAD';
if (BOUNDARY_ASSET_ID === 'PENDING_USER_UPLOAD') {
  throw new Error('Set BOUNDARY_ASSET_ID to the uploaded Earth Engine table asset ID.');
}

var boundaries = ee.FeatureCollection(BOUNDARY_ASSET_ID);
var expectedFeatureCount = 2316;
var scenarios = ee.List([
  'gctb_core_2021',
  'gctb_system_2021',
  'gub_2018',
  'ghs_uc_2020'
]);

var withGeometryType = boundaries.map(function(feature) {
  return feature.set('gee_geom_type', feature.geometry().type());
});
var raw = boundaries.filter(ee.Filter.eq('geom_var', 'raw_product_geometry'));
var nullUidCount = boundaries.size().subtract(
  boundaries.filter(ee.Filter.notNull(['uid'])).size()
);

var rawCityCounts = ee.Dictionary(scenarios.iterate(function(scenario, acc) {
  scenario = ee.String(scenario);
  var count = raw
    .filter(ee.Filter.eq('bnd_scn', scenario))
    .aggregate_count_distinct('city_id');
  return ee.Dictionary(acc).set(scenario, count);
}, ee.Dictionary({})));

print('Asset ID', BOUNDARY_ASSET_ID);
print('Feature count (expected 2316)', boundaries.size());
print('UID uniqueness (expected 2316)', boundaries.aggregate_count_distinct('uid'));
print('Scenario histogram', boundaries.aggregate_histogram('bnd_scn'));
print('Geometry variant histogram', boundaries.aggregate_histogram('geom_var'));
print('Geometry type histogram', withGeometryType.aggregate_histogram('gee_geom_type'));
print('Raw-geometry distinct city counts', rawCityCounts);
print('Null UID count', nullUidCount);
print('Non-positive boundary area count', boundaries.filter(ee.Filter.lte('bnd_area', 0)).size());

var audit = ee.Feature(null, {
  audit_run_id: 'gee_measurement_boundaries_20260805_r1',
  asset_id: BOUNDARY_ASSET_ID,
  expected_feature_count: expectedFeatureCount,
  observed_feature_count: boundaries.size(),
  observed_uid_unique_count: boundaries.aggregate_count_distinct('uid'),
  null_uid_count: nullUidCount,
  nonpositive_boundary_area_count: boundaries.filter(ee.Filter.lte('bnd_area', 0)).size(),
  gctb_core_raw_city_count: rawCityCounts.get('gctb_core_2021'),
  gctb_system_raw_city_count: rawCityCounts.get('gctb_system_2021'),
  gub_raw_city_count: rawCityCounts.get('gub_2018'),
  ghs_uc_raw_city_count: rawCityCounts.get('ghs_uc_2020'),
  local_manifest_sha256: '9ff9262e4ed74c0ddfe5a63f9c98a1072835d97837dae6a2e3caddef63a86de5'
});

Export.table.toDrive({
  collection: ee.FeatureCollection([audit]),
  description: 'R1_measurement_boundary_asset_audit',
  folder: 'GCTB_rebuild',
  fileNamePrefix: 'R1_measurement_boundary_asset_audit',
  fileFormat: 'CSV'
});
