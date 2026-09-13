package com.tibell.trafficml.model.definitionofsection;

import com.tibell.trafficml.entities.DefinitionOfSection;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * The {@code properties} object of a DefinitionOfSection GeoJSON Feature.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record DefinitionOfSectionPropertiesDto(
        String name,
        String region,
        Integer countyNo,
        LocalDateTime modifiedTime,
        LocalDateTime geometryModifiedTime,
        Boolean activated,
        LocalDateTime activatedModifiedTime,
        Integer averageFunctionalRoadClass,
        LocalDateTime importedTime) {
}
