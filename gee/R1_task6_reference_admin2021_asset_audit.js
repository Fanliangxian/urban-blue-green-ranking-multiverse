// Task 6A: audit the uploaded, repaired 2021 independent reference geometry.
var RUN_ID = 'task6_reference_admin2021_asset_audit_20260812_r1';
var ASSET_ID = 'projects/YOUR_EE_PROJECT/assets/GCTB/task6_reference_admin2021';
var EXPECTED_COUNT = 296;

var boundaries = ee.FeatureCollection(ASSET_ID);
var ids = ee.List(boundaries.aggregate_array('city_id'));
var sourceCodes = ee.List(boundaries.aggregate_array('src_code'));

print('Reference boundary count (expected 296)', boundaries.size());
print('Unique city_id count (expected 296)', ids.distinct().size());
print('Unique src_code count (expected 296)', sourceCodes.distinct().size());
print('Null city_id count (expected 0)',
  boundaries.size().subtract(boundaries.filter(ee.Filter.notNull(['city_id'])).size()));
print('Non-2021 source year count (expected 0)',
  boundaries.filter(ee.Filter.neq('src_year', 2021)).size());

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

var audited = boundaries.map(function(feature) {
  var rawGeometry = feature.geometry();
  var rawType = rawGeometry.type();
  var children = ee.List(ee.Algorithms.If(
    rawType.equals('GeometryCollection'), rawGeometry.geometries(), [rawGeometry]
  ));
  var childTypes = children.map(function(child) {
    return ee.Geometry(child).type();
  });
  var nonPolygonChildCount = childTypes.removeAll(['Polygon', 'MultiPolygon']).size();
  var geometry = polygonalGeometry(rawGeometry);
  var normalizedType = geometry.type();
  var rawArea = rawGeometry.area({maxError: 1});
  var area = geometry.area({maxError: 1});
  var perimeter = geometry.perimeter({maxError: 1});
  var equalAreaCirclePerimeter = ee.Number(2).multiply(
    ee.Number(Math.PI).multiply(area).sqrt()
  );
  var shapeComplexity = perimeter.divide(equalAreaCirclePerimeter).log();
  return ee.Feature(null, {
    run_id: RUN_ID,
    city_id: feature.get('city_id'),
    city_name: feature.get('city_name'),
    src_code: feature.get('src_code'),
    src_year: feature.get('src_year'),
    raw_geometry_type: rawType,
    child_geometry_types: childTypes.distinct().join('|'),
    non_polygon_child_count: nonPolygonChildCount,
    normalized_geometry_type: normalizedType,
    raw_area_m2: rawArea,
    area_m2: area,
    polygon_extraction_area_difference_m2: area.subtract(rawArea),
    polygon_extraction_relative_area_difference: area.subtract(rawArea).abs()
      .divide(rawArea.max(1)),
    perimeter_m: perimeter,
    shape_complexity_log_ratio: shapeComplexity,
    positive_area: area.gt(0),
    positive_perimeter: perimeter.gt(0)
  });
});

print('Raw geometry type histogram', audited.aggregate_histogram('raw_geometry_type'));
print('Raw GeometryCollection count (diagnostic)',
  audited.filter(ee.Filter.eq('raw_geometry_type', 'GeometryCollection')).size());
print('Raw GeometryCollection city IDs',
  audited.filter(ee.Filter.eq('raw_geometry_type', 'GeometryCollection'))
    .aggregate_array('city_id'));
print('Raw features containing a non-polygon child (diagnostic; observed 14)',
  audited.filter(ee.Filter.gt('non_polygon_child_count', 0)).size());
print('Child geometry-type combinations',
  audited.aggregate_histogram('child_geometry_types'));
print('Cities containing non-polygon children (diagnostic table)',
  audited.filter(ee.Filter.gt('non_polygon_child_count', 0))
    .select([
      'city_id', 'city_name', 'raw_geometry_type', 'child_geometry_types',
      'non_polygon_child_count'
    ]));
print('Normalized geometry type histogram',
  audited.aggregate_histogram('normalized_geometry_type'));
var polygonCount = audited.filter(
  ee.Filter.eq('normalized_geometry_type', 'Polygon')).size();
var multiPolygonCount = audited.filter(
  ee.Filter.eq('normalized_geometry_type', 'MultiPolygon')).size();
var polygonalCount = polygonCount.add(multiPolygonCount);
print('Normalized polygonal geometry count (expected 296)', polygonalCount);
print('Normalized non-polygonal geometry count (expected 0)',
  audited.size().subtract(polygonalCount));
print('Maximum polygon-extraction relative area difference (pass threshold <= 1e-10)',
  audited.aggregate_max('polygon_extraction_relative_area_difference'));
print('Non-positive area count (expected 0)',
  audited.filter(ee.Filter.eq('positive_area', false)).size());
print('Non-positive perimeter count (expected 0)',
  audited.filter(ee.Filter.eq('positive_perimeter', false)).size());
print('Audit preview', audited.limit(5));

Export.table.toDrive({
  collection: audited,
  description: 'R1_task6_reference_admin2021_asset_audit',
  folder: 'GCTB_rebuild',
  fileNamePrefix: 'R1_task6_reference_admin2021_asset_audit',
  fileFormat: 'CSV',
  selectors: [
    'run_id', 'city_id', 'city_name', 'src_code', 'src_year',
    'raw_geometry_type', 'child_geometry_types', 'non_polygon_child_count',
    'normalized_geometry_type', 'raw_area_m2', 'area_m2',
    'polygon_extraction_area_difference_m2',
    'polygon_extraction_relative_area_difference',
    'perimeter_m', 'shape_complexity_log_ratio',
    'positive_area', 'positive_perimeter'
  ]
});
