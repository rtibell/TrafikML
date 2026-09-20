package com.tibell.trafficml.services;

import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.model.speedofsection.MachineLearningSpeedOfSectionView;
import com.tibell.trafficml.repository.SpeedOfSectionRepository;
import com.tibell.trafficml.util.SwedishHolidays;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Locale;

import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

/**
 * Produces {@link MachineLearningSpeedOfSectionView} records for a section: its
 * speed/status history re-shaped into the numeric feature set (status/day-of-week/
 * holiday encodings, minutes since daybreak, ...) consumed by the traffic-speed ML
 * model.
 */
@Service
public class SpeedOfSectionMLData {

    private static final int MINUTES_PER_DAY = 24 * 60;
    private static final int DAYBREAK_MINUTE = 6 * 60;

    private static final int REGULAR_DAY = 0;
    private static final int EVE = 1;
    private static final int HOLIDAY = 2;

    private final SpeedOfSectionRepository repository;

    public SpeedOfSectionMLData(SpeedOfSectionRepository repository) {
        this.repository = repository;
    }

    public List<MachineLearningSpeedOfSectionView> forSection(Long sectionId, Pageable pageable) {
        return repository.findByIdSectionIdOrderByIdMeasureTimeDesc(sectionId, pageable).stream()
                .map(SpeedOfSectionMLData::toView)
                .toList();
    }

    private static MachineLearningSpeedOfSectionView toView(SpeedOfSection measurement) {
        LocalDateTime measureTime = measurement.getId().measureTime();
        LocalDate measureDate = measureTime.toLocalDate();
        String status = measurement.getStatus();
        int statusEnum = statusEnum(status);

        SwedishHolidays.HolidayResult nextHoliday = SwedishHolidays.nextHoliday(measureDate);
        int holidayType = holidayType(measureDate, nextHoliday);
        int holidayNum = nextHoliday.holidayNum;

        return new MachineLearningSpeedOfSectionView(
                measurement.getId().sectionId(),
                measureTime,
                status,
                statusEnum,
                measurement.getSpeed(),
                dayNr(measureTime),
                (int) nextHoliday.daysUntil,
                holidayType == REGULAR_DAY ? 1 : 0,
                holidayType == EVE ? 1 : 0,
                holidayType == HOLIDAY ? 1 : 0,
                statusEnum == Status.FREEFLOW.code ? 1 : 0,
                statusEnum == Status.HEAVY.code ? 1 : 0,
                statusEnum == Status.CONGESTED.code ? 1 : 0,
                statusEnum == Status.IMPOSSIBLE.code ? 1 : 0,
                minutesSinceDaybreak(measureTime),
                measureTime.getMonthValue(),
                holidayNum == SwedishHolidays.NYAR ? 1 : 0,
                holidayNum == SwedishHolidays.JUL ? 1 : 0,
                holidayNum == SwedishHolidays.FORSTA_MAJ ? 1 : 0,
                holidayNum == SwedishHolidays.NATIONALDAGEN ? 1 : 0,
                holidayNum == SwedishHolidays.PASK ? 1 : 0,
                holidayNum == SwedishHolidays.KRISTI_HIMMELSFARDSDAG ? 1 : 0,
                holidayNum == SwedishHolidays.PINGSTDAGEN ? 1 : 0,
                holidayNum == SwedishHolidays.MIDSOMMAR ? 1 : 0,
                holidayNum == SwedishHolidays.ALLA_HELGONS_DAG ? 1 : 0,
                holidayNum == SwedishHolidays.TRETTONDEDAG_JUL ? 1 : 0);
    }

    private static int statusEnum(String status) {
        if (status == null) {
            throw new IllegalStateException("Measurement has no status to map to statusEnum");
        }
        try {
            return Status.valueOf(status.toUpperCase(Locale.ROOT)).code;
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException("Unrecognized speed status: " + status, e);
        }
    }

    /** Monday = 0, ..., Sunday = 6. */
    private static int dayNr(LocalDateTime measureTime) {
        return measureTime.getDayOfWeek().getValue() - 1;
    }

    /**
     * Minutes elapsed since 06:00, wrapping around midnight so the "day" runs
     * 06:00-05:59 (e.g. 05:00 is 1380 minutes past the *previous* daybreak, not -60).
     */
    private static int minutesSinceDaybreak(LocalDateTime measureTime) {
        int minutesSinceMidnight = measureTime.getHour() * 60 + measureTime.getMinute();
        return Math.floorMod(minutesSinceMidnight - DAYBREAK_MINUTE, MINUTES_PER_DAY);
    }

    private static int holidayType(LocalDate measureDate, SwedishHolidays.HolidayResult nextHoliday) {
        if (isHoliday(measureDate)) {
            return HOLIDAY;
        }
        if (nextHoliday.daysUntil == 1) {
            return EVE;
        }
        return REGULAR_DAY;
    }

    private static boolean isHoliday(LocalDate measureDate) {
        return SwedishHolidays.nextHoliday(measureDate.minusDays(1)).daysUntil == 1;
    }

    private enum Status {
        FREEFLOW(0), HEAVY(1), CONGESTED(2), IMPOSSIBLE(3);

        private final int code;

        Status(int code) {
            this.code = code;
        }
    }
}
