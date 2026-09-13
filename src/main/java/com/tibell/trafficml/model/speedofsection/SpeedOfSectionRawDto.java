package com.tibell.trafficml.model.speedofsection;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * Raw JSON shape returned by {@code GET /api/traveltime}. {@code id} arrives as a JSON
 * string and is converted to {@code Long} by the mapper.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record SpeedOfSectionRawDto(
        String id,
        String status,
        Integer speed,
        LocalDateTime measureTime) {
}
