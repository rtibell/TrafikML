package com.tibell.trafficml.model.speedofsection;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.LocalDateTime;

/**
 * A single speed/status measurement re-shaped into the numeric feature set consumed by
 * the traffic-speed machine learning model. See {@code REST-opperations.md} for the
 * field mappings ({@code statusEnum}, {@code dayNr}, the {@code holiday_type_*} /
 * {@code *_status} / {@code holiday_is_*} one-hot encodings, ...).
 */
public record MachineLearningSpeedOfSectionView(
        Long sectionId,
        LocalDateTime measureTime,
        String status,
        int statusEnum,
        Integer speed,
        int dayNr,
        int daysUntilHoliday,
        @JsonProperty("holiday_type_regular_day") int holidayTypeRegularDay,
        @JsonProperty("holiday_type_eve") int holidayTypeEve,
        @JsonProperty("holiday_type_holiday") int holidayTypeHoliday,
        @JsonProperty("freeflow_status") int freeflowStatus,
        @JsonProperty("heavy_status") int heavyStatus,
        @JsonProperty("congested_status") int congestedStatus,
        @JsonProperty("imposible_status") int imposibleStatus,
        int minutesSincDaybreak,
        int monthOfYear,
        @JsonProperty("holiday_is_nyar") int holidayIsNyar,
        @JsonProperty("holiday_is_jul") int holidayIsJul,
        @JsonProperty("holiday_is_forsta_maj") int holidayIsForstaMaj,
        @JsonProperty("holiday_is_nationaldagen") int holidayIsNationaldagen,
        @JsonProperty("holiday_is_pask") int holidayIsPask,
        @JsonProperty("holiday_is_kristihimmelsfard") int holidayIsKristihimmelsfard,
        @JsonProperty("holiday_is_pingst") int holidayIsPingst,
        @JsonProperty("holiday_is_midsommar") int holidayIsMidsommar,
        @JsonProperty("holiday_is_allahelgona") int holidayIsAllahelgona,
        @JsonProperty("holiday_is_trettondag") int holidayIsTrettondag) {
}
