// GCTB 2021 candidate export for the frozen 296-city measurement workflow.
// This script does not use city names, 2022 GCTB anchors, nearest neighbours,
// or the old 291-city result to assign product entities.

var INPUT_ASSET =
    'projects/YOUR_EE_PROJECT/assets/GCTB/' +
    'B0_1_GCTB2021_China_candidates_raw';
var ADMIN_ASSET =
    'projects/YOUR_EE_PROJECT/assets/omega_adm_all_pref_shp';

var DRIVE_FOLDER = 'GCTB_rebuild';
var EXPORT_NAME = 'GCTB2021_296admin_candidates_r1';

var input = ee.FeatureCollection(INPUT_ASSET);
var adminRaw = ee.FeatureCollection(ADMIN_ASSET);

// Normalize the six-digit code without depending on its uploaded field type.
var adminWithCode = adminRaw.map(function(feature) {
  feature = ee.Feature(feature);
  var code6 = ee.Number.parse(
      ee.String(feature.get('city_code6')).slice(0, 6)
  ).format('%06d');
  return feature.set('code6_txt', code6);
});

// The local source contains prefecture cities, autonomous prefectures and
// directly administered county-level units.  A target city/municipality has a
// prefecture-level code ending 00 and a Chinese name ending 市.  Sansha has no
// source feature, so this returns the frozen 296 analysis assignment units.
var admin296 = adminWithCode.filter(ee.Filter.and(
    ee.Filter.stringEndsWith('code6_txt', '00'),
    ee.Filter.stringEndsWith('city_name', '市')
));

// Spatial filtering is candidate extraction only.  Exact positive-area
// intersections and all assignments are calculated locally in China Albers.
var candidates = input
    .filterBounds(admin296.geometry(100))
    .map(function(feature) {
      feature = ee.Feature(feature);
      return ee.Feature(feature.geometry(), {
        // Shapefile field names are limited to 10 characters.
        gctb_uid: ee.String(feature.get('gctb2021_id')),
        Area: feature.get('Area'),
        src_year: 2021,
        src_asset: 'B0_1_GCTB2021_China_candidates_raw',
        geom_type: feature.geometry().type()
      });
    });

// GCTB should represent polygon boundaries, but the frozen asset contains at
// least one non-polygon geometry.  Shapefile cannot mix geometry families.
// Under the preregistered positive-area rule, LineString/Point features are
// ineligible; they are quarantined rather than buffered or silently dropped.
var polygonTypeFilter = ee.Filter.inList(
    'geom_type', ['Polygon', 'MultiPolygon']
);
var polygonCandidates = candidates.filter(polygonTypeFilter);
var nonPolygonCandidates = candidates.filter(polygonTypeFilter.not());

var nonPolygonAudit = nonPolygonCandidates.map(function(feature) {
  feature = ee.Feature(feature);
  return ee.Feature(null, {
    gctb_uid: feature.get('gctb_uid'),
    Area: feature.get('Area'),
    src_year: feature.get('src_year'),
    src_asset: feature.get('src_asset'),
    geom_type: feature.get('geom_type'),
    exclusion_reason: 'non_polygon_zero_planar_area_ineligible'
  });
});

print('GCTB 2021 frozen candidate export');
print('Input candidate count:', input.size());
print('Raw admin feature count:', adminRaw.size());
print('Target admin count (MUST equal 296):', admin296.size());
print('Target admin code uniqueness:', admin296.aggregate_count_distinct('code6_txt'));
print('Export candidate count:', candidates.size());
print('First export feature:', candidates.first());
print('Export UID uniqueness:', candidates.aggregate_count_distinct('gctb_uid'));
print('Geometry type histogram:', candidates.aggregate_histogram('geom_type'));
print('Polygon export count:', polygonCandidates.size());
print('Polygon UID uniqueness:', polygonCandidates.aggregate_count_distinct('gctb_uid'));
print('Quarantined non-polygon count:', nonPolygonCandidates.size());
print('Quarantined non-polygon IDs:', nonPolygonCandidates.aggregate_array('gctb_uid'));

Map.centerObject(admin296, 4);
Map.addLayer(admin296.style({color: 'ff0000', fillColor: '00000000'}), {},
             'Frozen 296 assignment units');
Map.addLayer(polygonCandidates.limit(1000).style({color: 'ffd700', fillColor: '00000000'}),
             {}, 'GCTB 2021 candidates (first 1000)');

// maxVertices is deliberately omitted: setting it can cut one product entity
// into several exported records and violate stable entity identity.
Export.table.toDrive({
  collection: polygonCandidates,
  description: EXPORT_NAME,
  folder: DRIVE_FOLDER,
  fileNamePrefix: EXPORT_NAME,
  fileFormat: 'SHP',
  selectors: ['gctb_uid', 'Area', 'src_year', 'src_asset', 'geom_type']
});

// Attribute-only audit for every excluded non-polygon entity.
Export.table.toDrive({
  collection: nonPolygonAudit,
  description: EXPORT_NAME + '_nonpolygon_audit',
  folder: DRIVE_FOLDER,
  fileNamePrefix: EXPORT_NAME + '_nonpolygon_audit',
  fileFormat: 'CSV',
  selectors: [
    'gctb_uid', 'Area', 'src_year', 'src_asset', 'geom_type',
    'exclusion_reason'
  ]
});
