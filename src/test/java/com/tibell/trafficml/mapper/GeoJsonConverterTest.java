package com.tibell.trafficml.mapper;

import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.model.PointGeoJson;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.locationtech.jts.geom.LineString;
import org.locationtech.jts.geom.Point;

class GeoJsonConverterTest {

    @Test
    void toPointConvertsLongitudeLatitudeAndSrid() {
        PointGeoJson geoJson = new PointGeoJson("Point", new double[] { 18.0967, 59.5750 });

        Point point = GeoJsonConverter.toPoint(geoJson, GeoJsonConverter.SRID_WGS84);

        assertThat(point.getX()).isEqualTo(18.0967);
        assertThat(point.getY()).isEqualTo(59.5750);
        assertThat(point.getSRID()).isEqualTo(4326);
    }

    @Test
    void toPointReturnsNullForNullInput() {
        assertThat(GeoJsonConverter.toPoint(null, GeoJsonConverter.SRID_WGS84)).isNull();
    }

    @Test
    void toLineStringPreservesCoordinateOrderAndSrid() {
        LineStringGeoJson geoJson = new LineStringGeoJson("LineString", new double[][] {
                { 17.910571, 58.972645 },
                { 17.911648, 58.974388 },
                { 17.914212, 58.977026 }
        });

        LineString lineString = GeoJsonConverter.toLineString(geoJson, GeoJsonConverter.SRID_SWEREF99_TM);

        assertThat(lineString.getNumPoints()).isEqualTo(3);
        assertThat(lineString.getCoordinateN(0).x).isEqualTo(17.910571);
        assertThat(lineString.getCoordinateN(0).y).isEqualTo(58.972645);
        assertThat(lineString.getCoordinateN(2).x).isEqualTo(17.914212);
        assertThat(lineString.getSRID()).isEqualTo(3006);
    }

    @Test
    void toLineStringReturnsNullForNullInput() {
        assertThat(GeoJsonConverter.toLineString(null, GeoJsonConverter.SRID_WGS84)).isNull();
    }
}
