package com.tibell.trafficml.model.speedofsection;

import java.time.LocalDateTime;

/**
 * A single speed/status measurement re-shaped into the numeric feature set consumed by
 * the traffic-speed machine learning model. See {@code REST-opperations.md} for the
 * field mappings ({@code statusEnum}, {@code dayNr}, {@code holidayType}, ...).
 */
public record MachineLearningSpeedOfSectionView(
        Long sectionId,
        LocalDateTime measureTime,
        String status,
        int statusEnum,
        Integer speed,
        int dayNr,
        int daysUntilHoliday,
        int holidayType,
        int minutesSincDaybreak,
        int monthOfYear,
        int holidayNum) {
}
