package com.tibell.trafficml.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * Minimal GeoJSON {@code LineString} as returned by trafiken.nu, e.g.
 * {@code {"type": "LineString", "coordinates": [[lon, lat], [lon, lat], ...]}}.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record LineStringGeoJson(String type, double[][] coordinates) {
}
