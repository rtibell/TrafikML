package com.tibell.trafficml.util;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.Year;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import tools.jackson.databind.json.JsonMapper;

import com.tibell.trafficml.util.SwedishHolidays.HolidayResult;

/**
 * Generates a full-year report of {@link SwedishHolidays#nextHoliday(LocalDate)}'s
 * output, one entry per calendar date of 2026, and writes it as JSON to
 * {@code build/NextHoliday-2026.log}.
 */
class SwedishHolidaysYearReportTest {

    private static final int YEAR = 2026;
    private static final Path REPORT_FILE = Path.of("build", "NextHoliday-2026.log");

    private record DailyHoliday(
            LocalDate date,
            String nextHolidayName,
            LocalDate nextHolidayDate,
            long daysUntilHoliday,
            int holidayNum) {

        static DailyHoliday of(LocalDate date) {
            HolidayResult result = SwedishHolidays.nextHoliday(date);
            return new DailyHoliday(date, result.name, result.date, result.daysUntil, result.holidayNum);
        }
    }

    @Test
    void writesNextHolidayForEveryDateOfTheYear() throws IOException {
        List<DailyHoliday> report = new ArrayList<>();
        LocalDate date = LocalDate.of(YEAR, 1, 1);
        LocalDate endOfYear = LocalDate.of(YEAR, 12, 31);
        while (!date.isAfter(endOfYear)) {
            report.add(DailyHoliday.of(date));
            date = date.plusDays(1);
        }

        Files.createDirectories(REPORT_FILE.getParent());
        JsonMapper.builder().build()
                .writerWithDefaultPrettyPrinter()
                .writeValue(REPORT_FILE.toFile(), report);

        assertThat(report).hasSize(Year.of(YEAR).length());
        assertThat(Files.exists(REPORT_FILE)).isTrue();

        DailyHoliday first = report.get(0);
        assertThat(first.date()).isEqualTo(LocalDate.of(2026, 1, 1));
        assertThat(first.nextHolidayName()).isEqualTo("Trettondag jul");
        assertThat(first.daysUntilHoliday()).isEqualTo(5);

        DailyHoliday last = report.get(report.size() - 1);
        assertThat(last.date()).isEqualTo(LocalDate.of(2026, 12, 31));
        assertThat(last.nextHolidayName()).isEqualTo("Nyårsdagen");
        assertThat(last.nextHolidayDate()).isEqualTo(LocalDate.of(2027, 1, 1));
        assertThat(last.daysUntilHoliday()).isEqualTo(1);
    }
}
