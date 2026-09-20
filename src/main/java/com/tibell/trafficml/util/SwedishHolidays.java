package com.tibell.trafficml.util;

import java.time.DayOfWeek;
import java.time.Duration;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Computes the next Swedish public holiday on or after a given date. Used by
 * {@link com.tibell.trafficml.services.SpeedOfSectionMLData} to derive the
 * {@code daysUntilHoliday}, {@code holidayType}, and {@code holidayNum} machine-learning
 * features.
 */
public class SwedishHolidays {

    // -----------------------------
    // Public API
    // -----------------------------
    public static HolidayResult nextHoliday(LocalDate fromDate) {
        int year = fromDate.getYear();

        List<Holiday> holidays = generateHolidays(year);

        // Filter holidays after the given date
        Holiday next = holidays.stream()
                .filter(h -> h.date.isAfter(fromDate))
                .sorted(Comparator.comparing(h -> h.date))
                .findFirst()
                .orElse(null);

        // If none left in this year → look at next year
        if (next == null) {
            holidays = generateHolidays(year + 1);
            next = holidays.stream()
                    .sorted(Comparator.comparing(h -> h.date))
                    .findFirst()
                    .orElseThrow();
        }

        long daysUntil = Duration.between(fromDate.atStartOfDay(), next.date.atStartOfDay()).toDays();

        return new HolidayResult(next.name, next.date, daysUntil, next.holidayNum);
    }

    // -----------------------------
    // Predefined holiday numbers (holidayNum): a stable identifier for which holiday
    // is being referred to, independent of the year it falls in.
    // -----------------------------
    public static final int NYAR = 1;
    public static final int JUL = 2;
    public static final int TRETTONDEDAG_JUL = 11;
    public static final int FORSTA_MAJ = 3;
    public static final int NATIONALDAGEN = 4;
    public static final int PASK = 6;
    public static final int KRISTI_HIMMELSFARDSDAG = 7;
    public static final int PINGSTDAGEN = 8;
    public static final int MIDSOMMAR = 9;
    public static final int ALLA_HELGONS_DAG = 10;
//    private static final int NYARSDAGEN = 1;
//    private static final int TRETTONDEDAG_JUL = 2;
//    private static final int JULDAGEN = 5;
//    private static final int ANNANDAG_JUL = 6;
//    private static final int LANGFREDAGEN = 7;
//    private static final int PASKAFTON = 8;
//    private static final int PASKDAGEN = 9;
//    private static final int ANNANDAG_PASK = 10;
//    private static final int KRISTI_HIMMELSFARDSDAG = 11;
//    private static final int PINGSTDAGEN = 12;
//    private static final int MIDSOMMARDAGEN = 13;
//    private static final int ALLA_HELGONS_DAG = 14;

    // -----------------------------
    // Holiday generation
    // -----------------------------
    private static List<Holiday> generateHolidays(int year) {
        List<Holiday> list = new ArrayList<>();

        // Fixed holidays
        list.add(new Holiday("Nyårsdagen", LocalDate.of(year, 1, 1), NYAR));
//        list.add(new Holiday("Nyårsdagen", LocalDate.of(year, 1, 1), NYARSDAGEN));
        list.add(new Holiday("Trettondag jul", LocalDate.of(year, 1, 6), TRETTONDEDAG_JUL));
//        list.add(new Holiday("Jul", LocalDate.of(year, 1, 6), TRETTONDEDAG_JUL));
        list.add(new Holiday("Första maj", LocalDate.of(year, 5, 1), FORSTA_MAJ));
        list.add(new Holiday("Nationaldagen", LocalDate.of(year, 6, 6), NATIONALDAGEN));
//        list.add(new Holiday("Juldagen", LocalDate.of(year, 12, 25), JULDAGEN));
//        list.add(new Holiday("Annandag jul", LocalDate.of(year, 12, 26), ANNANDAG_JUL));
        list.add(new Holiday("Jul", LocalDate.of(year, 12, 26), JUL));

        // Easter-based holidays
        LocalDate easter = calculateEasterSunday(year);
//        list.add(new Holiday("Långfredagen", easter.minusDays(2), LANGFREDAGEN));
//        list.add(new Holiday("Påskafton", easter.minusDays(1), PASKAFTON));
//        list.add(new Holiday("Påskdagen", easter, PASKDAGEN));
//        list.add(new Holiday("Annandag påsk", easter.plusDays(1), ANNANDAG_PASK));
        list.add(new Holiday("Påsk", easter.plusDays(1), PASK));

        list.add(new Holiday("Kristi himmelsfärdsdag", easter.plusDays(39), KRISTI_HIMMELSFARDSDAG));
        list.add(new Holiday("Pingstdagen", easter.plusDays(49), PINGSTDAGEN));

        // Weekday-based holidays
        list.add(new Holiday("Midsommardagen", midsummerDay(year), MIDSOMMAR));
        list.add(new Holiday("Alla helgons dag", allSaintsDay(year), ALLA_HELGONS_DAG));

        return list;
    }

    // -----------------------------
    // Easter calculation (Meeus/Jones/Butcher)
    // -----------------------------
    private static LocalDate calculateEasterSunday(int year) {
        int a = year % 19;
        int b = year / 100;
        int c = year % 100;
        int d = b / 4;
        int e = b % 4;
        int f = (b + 8) / 25;
        int g = (b - f + 1) / 3;
        int h = (19 * a + b - d - g + 15) % 30;
        int i = c / 4;
        int k = c % 4;
        int L = (32 + 2 * e + 2 * i - h - k) % 7;
        int m = (a + 11 * h + 22 * L) / 451;
        int month = (h + L - 7 * m + 114) / 31;
        int day = ((h + L - 7 * m + 114) % 31) + 1;

        return LocalDate.of(year, month, day);
    }

    // -----------------------------
    // Midsummer Day (Saturday 20–26 June)
    // -----------------------------
    private static LocalDate midsummerDay(int year) {
        LocalDate date = LocalDate.of(year, 6, 20);
        while (date.getDayOfWeek() != DayOfWeek.SATURDAY) {
            date = date.plusDays(1);
        }
        return date;
    }

    // -----------------------------
    // All Saints Day (Saturday 31 Oct–6 Nov)
    // -----------------------------
    private static LocalDate allSaintsDay(int year) {
        LocalDate date = LocalDate.of(year, 10, 31);
        while (date.getDayOfWeek() != DayOfWeek.SATURDAY) {
            date = date.plusDays(1);
        }
        return date;
    }

    // -----------------------------
    // Data classes
    // -----------------------------
    private static class Holiday {
        String name;
        LocalDate date;
        int holidayNum;

        Holiday(String name, LocalDate date, int holidayNum) {
            this.name = name;
            this.date = date;
            this.holidayNum = holidayNum;
        }
    }

    public static class HolidayResult {
        public final String name;
        public final LocalDate date;
        public final long daysUntil;
        public final int holidayNum;

        HolidayResult(String name, LocalDate date, long daysUntil, int holidayNum) {
            this.name = name;
            this.date = date;
            this.daysUntil = daysUntil;
            this.holidayNum = holidayNum;
        }

        @Override
        public String toString() {
            return name + " infaller " + date + " (om " + daysUntil + " dagar)";
        }
    }
}
