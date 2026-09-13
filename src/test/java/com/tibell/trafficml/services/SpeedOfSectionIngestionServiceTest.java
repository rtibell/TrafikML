package com.tibell.trafficml.services;

import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.entities.SpeedOfSectionId;
import com.tibell.trafficml.mapper.SpeedOfSectionMapper;
import com.tibell.trafficml.model.speedofsection.SpeedOfSectionRawDto;
import com.tibell.trafficml.repository.SpeedOfSectionRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.repository.DefinitionOfSectionRepository;

@ExtendWith(MockitoExtension.class)
class SpeedOfSectionIngestionServiceTest {

    @Mock
    private SpeedOfSectionRepository repository;

    @Mock
    private DefinitionOfSectionRepository sectionRepository;

    private final SpeedOfSectionMapper mapper = new SpeedOfSectionMapper();
    private SpeedOfSectionIngestionService service;

    @BeforeEach
    void setUp() {
        service = new SpeedOfSectionIngestionService(repository, sectionRepository, mapper);
    }

    @Test
    void insertsNewMeasurementWhenSectionIsKnown() {
        LocalDateTime measureTime = LocalDateTime.parse("2026-09-12T17:19:00");
        SpeedOfSectionRawDto dto = new SpeedOfSectionRawDto("44469", "freeflow", 48, measureTime);
        SpeedOfSectionId id = new SpeedOfSectionId(44469L, measureTime);
        when(repository.findById(id)).thenReturn(Optional.empty());
        when(sectionRepository.findById(44469L)).thenReturn(Optional.of(new DefinitionOfSection(44469L)));

        var result = service.ingest(List.of(dto));

        assertThat(result.inserted()).isEqualTo(1);
        assertThat(result.updated()).isZero();
        assertThat(result.skippedUnknownSection()).isZero();
        verify(repository).save(any(SpeedOfSection.class));
    }

    @Test
    void skipsMeasurementWhenSectionIsUnknown() {
        LocalDateTime measureTime = LocalDateTime.now();
        SpeedOfSectionRawDto dto = new SpeedOfSectionRawDto("999", "freeflow", 48, measureTime);
        when(repository.findById(new SpeedOfSectionId(999L, measureTime))).thenReturn(Optional.empty());
        when(sectionRepository.findById(999L)).thenReturn(Optional.empty());

        var result = service.ingest(List.of(dto));

        assertThat(result.skippedUnknownSection()).isEqualTo(1);
        assertThat(result.inserted()).isZero();
        verify(repository, never()).save(any());
    }

    @Test
    void updatesExistingMeasurementForSameCompositeKey() {
        LocalDateTime measureTime = LocalDateTime.now();
        DefinitionOfSection section = new DefinitionOfSection(1L);
        SpeedOfSection existing = mapper.toNewEntity(section, new SpeedOfSectionRawDto("1", "freeflow", 60, measureTime));
        when(repository.findById(new SpeedOfSectionId(1L, measureTime))).thenReturn(Optional.of(existing));

        SpeedOfSectionRawDto updatedDto = new SpeedOfSectionRawDto("1", "congested", 10, measureTime);
        var result = service.ingest(List.of(updatedDto));

        assertThat(result.updated()).isEqualTo(1);
        assertThat(existing.getStatus()).isEqualTo("congested");
        assertThat(existing.getSpeed()).isEqualTo(10);
        assertThat(existing.getRecordUpdated()).isNotNull();
        verify(sectionRepository, never()).findById(any());
    }
}
