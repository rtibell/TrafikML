package com.tibell.trafficml.model.definitionofsection;

import com.tibell.trafficml.entities.DefinitionOfSection;

import java.time.LocalDateTime;

/**
 * Read-only JSON projection of {@link DefinitionOfSection} exposed via the REST API.
 * The JTS {@code geometry} is rendered as a flat coordinate array to keep the payload
 * plain JSON (no vendor-specific WKT/WKB leaking to clients).
 */
public record DefinitionOfSectionView(
        Long id,
        String name,
        String region,
        Integer countyNo,
        LocalDateTime modifiedTime,
        LocalDateTime geometryModifiedTime,
        Boolean activated,
        Integer averageFunctionalRoadClass,
        double[][] coordinates) {

    public static DefinitionOfSectionView from(DefinitionOfSection entity) {
        double[][] coordinates = entity.getGeometry() == null
                ? new double[0][]
                : java.util.Arrays.stream(entity.getGeometry().getCoordinates())
                        .map(c -> new double[] { c.x, c.y })
                        .toArray(double[][]::new);

        return new DefinitionOfSectionView(
                entity.getId(),
                entity.getName(),
                entity.getRegion(),
                entity.getCountyNo(),
                entity.getModifiedTime(),
                entity.getGeometryModifiedTime(),
                entity.getActivated(),
                entity.getAverageFunctionalRoadClass(),
                coordinates);
    }
}
