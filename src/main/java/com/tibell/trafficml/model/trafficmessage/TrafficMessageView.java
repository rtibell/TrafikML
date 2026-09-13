package com.tibell.trafficml.model.trafficmessage;

import com.tibell.trafficml.entities.TrafficMessage;

import java.time.LocalDateTime;

public record TrafficMessageView(
        Long id,
        String region,
        String title,
        String message,
        String messageType,
        String trafficType,
        String affectedDirection,
        LocalDateTime startTime,
        LocalDateTime endTime,
        Integer severity,
        String severityText,
        Boolean roadClosed,
        String detailsPath) {

    public static TrafficMessageView from(TrafficMessage entity) {
        return new TrafficMessageView(
                entity.getId(),
                entity.getRegion(),
                entity.getTitle(),
                entity.getMessage(),
                entity.getMessageType(),
                entity.getTrafficType(),
                entity.getAffectedDirection(),
                entity.getStartTime(),
                entity.getEndTime(),
                entity.getSeverity(),
                entity.getSeverityText(),
                entity.getRoadClosed(),
                entity.getDetailsPath());
    }
}
