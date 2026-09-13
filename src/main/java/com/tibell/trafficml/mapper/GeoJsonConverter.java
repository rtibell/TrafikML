package com.tibell.trafficml.mapper;

import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.model.PointGeoJson;

import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.LineString;
import org.locationtech.jts.geom.PrecisionModel;
import org.locationtech.jts.geom.Point;

/**
 * Converts the trafiken.nu GeoJSON DTOs into JTS geometries that Hibernate Spatial can
 * persist to PostGIS {@code geometry} columns.
 */
public final class GeoJsonConverter {

    /** WGS84 (longitude/latitude), used for {@code geometry}/{@code wgs84Position}. */
    public static final int SRID_WGS84 = 4326;

    /** SWEREF 99 TM, used for {@code sweRef99Extent}. */
    public static final int SRID_SWEREF99_TM = 3006;

    private GeoJsonConverter() {
    }

    public static Point toPoint(PointGeoJson geoJson, int srid) {
        if (geoJson == null) {
            return null;
        }
        GeometryFactory factory = factory(srid);
        return factory.createPoint(new Coordinate(geoJson.longitude(), geoJson.latitude()));
    }

    public static LineString toLineString(LineStringGeoJson geoJson, int srid) {
        if (geoJson == null || geoJson.coordinates() == null) {
            return null;
        }
        Coordinate[] coordinates = new Coordinate[geoJson.coordinates().length];
        for (int i = 0; i < geoJson.coordinates().length; i++) {
            double[] pair = geoJson.coordinates()[i];
            coordinates[i] = new Coordinate(pair[0], pair[1]);
        }
        return factory(srid).createLineString(coordinates);
    }

    private static GeometryFactory factory(int srid) {
        return new GeometryFactory(new PrecisionModel(), srid);
    }
}
