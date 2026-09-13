package com.tibell.trafficml.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * Minimal GeoJSON {@code Point} as returned by trafiken.nu, e.g.
 * {@code {"type": "Point", "coordinates": [18.0967, 59.5750]}}.
 * Coordinates are {@code [longitude, latitude]} per the GeoJSON spec.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record PointGeoJson(String type, double[] coordinates) {

    public double longitude() {
        return coordinates[0];
    }

    public double latitude() {
        return coordinates[1];
    }
}
