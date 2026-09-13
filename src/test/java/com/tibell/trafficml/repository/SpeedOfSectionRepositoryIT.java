package com.tibell.trafficml.repository;

import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.entities.SpeedOfSectionId;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.time.LocalDateTime;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

import jakarta.persistence.EntityManager;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.repository.DefinitionOfSectionRepository;
import com.tibell.trafficml.testsupport.AbstractIntegrationTest;

class SpeedOfSectionRepositoryIT extends AbstractIntegrationTest {

    @Autowired
    private SpeedOfSectionRepository repository;

    @Autowired
    private DefinitionOfSectionRepository sectionRepository;

    @Autowired
    private EntityManager entityManager;

    @Test
    void savesAndReloadsMeasurementLinkedToItsSection() {
        DefinitionOfSection section = new DefinitionOfSection(999002L);
        section.setModifiedTime(LocalDateTime.now());
        section.setGeometryModifiedTime(LocalDateTime.now());
        section.setActivatedModifiedTime(LocalDateTime.now());
        section.setRecordCreated(Instant.now());
        sectionRepository.saveAndFlush(section);

        LocalDateTime measureTime = LocalDateTime.parse("2026-09-12T17:19:00");
        SpeedOfSection measurement = new SpeedOfSection(section, measureTime);
        measurement.setStatus("freeflow");
        measurement.setSpeed(48);
        measurement.setRecordCreated(Instant.now());
        repository.saveAndFlush(measurement);
        entityManager.clear();

        SpeedOfSectionId id = new SpeedOfSectionId(999002L, measureTime);
        SpeedOfSection reloaded = repository.findById(id).orElseThrow();
        assertThat(reloaded.getStatus()).isEqualTo("freeflow");
        assertThat(reloaded.getSpeed()).isEqualTo(48);
        assertThat(reloaded.getSection().getId()).isEqualTo(999002L);
    }

    @Test
    void twoMeasurementsForSameSectionAtDifferentTimesAreBothStored() {
        DefinitionOfSection section = new DefinitionOfSection(999003L);
        section.setModifiedTime(LocalDateTime.now());
        section.setGeometryModifiedTime(LocalDateTime.now());
        section.setActivatedModifiedTime(LocalDateTime.now());
        section.setRecordCreated(Instant.now());
        sectionRepository.saveAndFlush(section);

        SpeedOfSection first = new SpeedOfSection(section, LocalDateTime.parse("2026-09-12T17:19:00"));
        first.setSpeed(48);
        first.setRecordCreated(Instant.now());
        SpeedOfSection second = new SpeedOfSection(section, LocalDateTime.parse("2026-09-12T17:20:00"));
        second.setSpeed(52);
        second.setRecordCreated(Instant.now());

        repository.saveAllAndFlush(java.util.List.of(first, second));

        assertThat(repository.findByIdSectionIdOrderByIdMeasureTimeDesc(999003L,
                org.springframework.data.domain.PageRequest.of(0, 10))).hasSize(2);
    }
}
