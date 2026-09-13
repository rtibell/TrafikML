package com.tibell.trafficml.model.speedofsection;

import com.tibell.trafficml.entities.SpeedOfSection;

import java.time.LocalDateTime;

public record SpeedOfSectionView(Long sectionId, LocalDateTime measureTime, String status, Integer speed) {

    public static SpeedOfSectionView from(SpeedOfSection entity) {
        return new SpeedOfSectionView(
                entity.getId().sectionId(),
                entity.getId().measureTime(),
                entity.getStatus(),
                entity.getSpeed());
    }
}
