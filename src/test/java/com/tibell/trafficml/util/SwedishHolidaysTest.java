package com.tibell.trafficml.util;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDate;

import org.junit.jupiter.api.Test;

import com.tibell.trafficml.util.SwedishHolidays.HolidayResult;

class SwedishHolidaysTest {

    @Test
    void findsFixedDateHolidayLaterInTheSameYear() {
        HolidayResult result = SwedishHolidays.nextHoliday(LocalDate.of(2026, 1, 1));

        assertThat(result.name).isEqualTo("Trettondag jul");
        assertThat(result.date).isEqualTo(LocalDate.of(2026, 1, 6));
        assertThat(result.daysUntil).isEqualTo(5);
        assertThat(result.holidayNum).isEqualTo(11);
    }

    @Test
    void isStrictlyAfterTheGivenDate() {
        // 2026-01-01 is itself Nyårsdagen; nextHoliday must not return that same day.
        HolidayResult result = SwedishHolidays.nextHoliday(LocalDate.of(2026, 1, 1));

        assertThat(result.date).isNotEqualTo(LocalDate.of(2026, 1, 1));
    }

    @Test
    void rollsOverIntoNextYearAtYearEnd() {
        HolidayResult result = SwedishHolidays.nextHoliday(LocalDate.of(2026, 12, 31));

        assertThat(result.name).isEqualTo("Nyårsdagen");
        assertThat(result.date).isEqualTo(LocalDate.of(2027, 1, 1));
        assertThat(result.daysUntil).isEqualTo(1);
        assertThat(result.holidayNum).isEqualTo(1);
    }

    @Test
    void findsEasterBasedHoliday() {
        HolidayResult result = SwedishHolidays.nextHoliday(LocalDate.of(2026, 4, 2));

        assertThat(result.name).isEqualTo("Påsk");
        assertThat(result.date).isEqualTo(LocalDate.of(2026, 4, 6));
        assertThat(result.holidayNum).isEqualTo(6);
    }

    @Test
    void findsWeekdayBasedHoliday() {
        HolidayResult midsummer = SwedishHolidays.nextHoliday(LocalDate.of(2026, 6, 19));
        assertThat(midsummer.name).isEqualTo("Midsommardagen");
        assertThat(midsummer.date).isEqualTo(LocalDate.of(2026, 6, 20));
        assertThat(midsummer.holidayNum).isEqualTo(9);

        HolidayResult allSaints = SwedishHolidays.nextHoliday(LocalDate.of(2026, 10, 30));
        assertThat(allSaints.name).isEqualTo("Alla helgons dag");
        assertThat(allSaints.date).isEqualTo(LocalDate.of(2026, 10, 31));
        assertThat(allSaints.holidayNum).isEqualTo(10);
    }

    @Test
    void holidayNumIsStableAcrossYears() {
        HolidayResult newYear2026 = SwedishHolidays.nextHoliday(LocalDate.of(2025, 12, 31));
        HolidayResult newYear2027 = SwedishHolidays.nextHoliday(LocalDate.of(2026, 12, 31));

        assertThat(newYear2026.holidayNum).isEqualTo(newYear2027.holidayNum);
        assertThat(newYear2026.date.getYear()).isNotEqualTo(newYear2027.date.getYear());
    }
}
