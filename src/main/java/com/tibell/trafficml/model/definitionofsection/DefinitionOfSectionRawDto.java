package com.tibell.trafficml.model.definitionofsection;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import com.tibell.trafficml.model.LineStringGeoJson;

/**
 * Raw JSON shape returned by {@code GET /api/traveltime/action/getsections}: a GeoJSON
 * {@code Feature} per road section. {@code id} arrives as a JSON string and is converted
 * to {@code Long} by the mapper.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record DefinitionOfSectionRawDto(
        String type,
        String id,
        LineStringGeoJson geometry,
        DefinitionOfSectionPropertiesDto properties) {
}
