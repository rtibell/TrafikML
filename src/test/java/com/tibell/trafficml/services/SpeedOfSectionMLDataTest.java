package com.tibell.trafficml.services;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.model.speedofsection.MachineLearningSpeedOfSectionView;
import com.tibell.trafficml.repository.SpeedOfSectionRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.Pageable;

@ExtendWith(MockitoExtension.class)
class SpeedOfSectionMLDataTest {

    @Mock
    private SpeedOfSectionRepository repository;

    private SpeedOfSectionMLData mlData;

    @BeforeEach
    void setUp() {
        mlData = new SpeedOfSectionMLData(repository);
    }

    @Test
    void mapsARegularWeekdayMeasurement() {
        // Wednesday 2026-07-15, safely between Midsummer and Alla helgons dag: no
        // Swedish holiday within a day of it in either direction.
        MachineLearningSpeedOfSectionView view = fetchSingle(44469L, "2026-07-15T13:44:00", "freeflow", 70);

        assertThat(view.sectionId()).isEqualTo(44469L);
        assertThat(view.measureTime()).isEqualTo(LocalDateTime.parse("2026-07-15T13:44:00"));
        assertThat(view.status()).isEqualTo("freeflow");
        assertThat(view.statusEnum()).isEqualTo(0);
        assertThat(view.speed()).isEqualTo(70);
        assertThat(view.dayNr()).isEqualTo(2); // Wednesday
        assertThat(view.monthOfYear()).isEqualTo(7);
        assertThat(view.minutesSincDaybreak()).isEqualTo(464); // 13:44 - 06:00
        assertThat(view.daysUntilHoliday()).isEqualTo(108); // -> Alla helgons dag, 2026-10-31
        assertThat(view.holidayType()).isEqualTo(0); // regular day
        assertThat(view.holidayNum()).isEqualTo(10); // Alla helgons dag
    }

    @Test
    void mapsAllFourStatusValues() {
        assertThat(fetchSingle(1L, "2026-07-15T12:00:00", "freeflow", 70).statusEnum()).isEqualTo(0);
        assertThat(fetchSingle(1L, "2026-07-15T12:00:00", "heavy", 40).statusEnum()).isEqualTo(1);
        assertThat(fetchSingle(1L, "2026-07-15T12:00:00", "congested", 15).statusEnum()).isEqualTo(2);
        assertThat(fetchSingle(1L, "2026-07-15T12:00:00", "impossible", 0).statusEnum()).isEqualTo(3);
    }

    @Test
    void unrecognizedStatusThrows() {
        DefinitionOfSection section = new DefinitionOfSection(1L);
        SpeedOfSection measurement = new SpeedOfSection(section, LocalDateTime.parse("2026-07-15T12:00:00"));
        measurement.setStatus("unknown");
        when(repository.findByIdSectionIdOrderByIdMeasureTimeDesc(eq(1L), any(Pageable.class)))
                .thenReturn(List.of(measurement));

        assertThatThrownBy(() -> mlData.forSection(1L, Pageable.unpaged()))
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void newYearsDayIsAHoliday() {
        MachineLearningSpeedOfSectionView view = fetchSingle(1L, "2026-01-01T10:00:00", "freeflow", 70);

        assertThat(view.holidayType()).isEqualTo(2); // holiday
        assertThat(view.daysUntilHoliday()).isEqualTo(5); // -> Trettondedag jul, 2026-01-06
        assertThat(view.holidayNum()).isEqualTo(11); // Trettondedag jul
    }

    @Test
    void newYearsEveIsTheEveOfAHoliday() {
        MachineLearningSpeedOfSectionView view = fetchSingle(1L, "2025-12-31T22:00:00", "freeflow", 70);

        assertThat(view.holidayType()).isEqualTo(1); // eve
        assertThat(view.daysUntilHoliday()).isEqualTo(1); // -> Nyårsdagen, 2026-01-01
        assertThat(view.holidayNum()).isEqualTo(1); // Nyårsdagen
    }

    @Test
    void minutesSinceDaybreakWrapsAroundMidnight() {
        assertThat(fetchSingle(1L, "2026-07-15T06:00:00", "freeflow", 70).minutesSincDaybreak()).isEqualTo(0);
        assertThat(fetchSingle(1L, "2026-07-15T05:00:00", "freeflow", 70).minutesSincDaybreak()).isEqualTo(1380);
    }

    private MachineLearningSpeedOfSectionView fetchSingle(Long sectionId, String measureTime, String status, int speed) {
        DefinitionOfSection section = new DefinitionOfSection(sectionId);
        SpeedOfSection measurement = new SpeedOfSection(section, LocalDateTime.parse(measureTime));
        measurement.setStatus(status);
        measurement.setSpeed(speed);
        when(repository.findByIdSectionIdOrderByIdMeasureTimeDesc(eq(sectionId), any(Pageable.class)))
                .thenReturn(List.of(measurement));

        List<MachineLearningSpeedOfSectionView> views = mlData.forSection(sectionId, Pageable.unpaged());
        assertThat(views).hasSize(1);
        return views.get(0);
    }
}
