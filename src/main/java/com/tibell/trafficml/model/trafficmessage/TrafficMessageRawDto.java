package com.tibell.trafficml.model.trafficmessage;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.model.PointGeoJson;

/**
 * Raw JSON shape returned by {@code GET /api/trafficmessages}. Unlike the other two
 * feeds, {@code id} already arrives as a JSON number.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record TrafficMessageRawDto(
        Long id,
        String region,
        String title,
        String message,
        String provider,
        String messageType,
        String messageCode,
        String trafficType,
        PointGeoJson wgs84Position,
        LineStringGeoJson sweRef99Extent,
        String affectedDirection,
        LocalDateTime startTime,
        LocalDateTime endTime,
        List<Map<String, Object>> scheduledOccurrences,
        LocalDateTime versionTime,
        Integer iconId,
        Integer severity,
        Boolean isFuture,
        Boolean roadClosed,
        String detailsPath,
        Integer sortIndex,
        String severityText,
        Integer workState) {
}
