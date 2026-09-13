package com.tibell.trafficml.services;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.mapper.DefinitionOfSectionMapper;
import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionPropertiesDto;
import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionRawDto;
import com.tibell.trafficml.repository.DefinitionOfSectionRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
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

import com.tibell.trafficml.model.LineStringGeoJson;

@ExtendWith(MockitoExtension.class)
class DefinitionOfSectionIngestionServiceTest {

    @Mock
    private DefinitionOfSectionRepository repository;

    private final DefinitionOfSectionMapper mapper = new DefinitionOfSectionMapper();
    private DefinitionOfSectionIngestionService service;

    @BeforeEach
    void setUp() {
        service = new DefinitionOfSectionIngestionService(repository, mapper);
    }

    @Test
    void insertsSectionUnseenBefore() {
        when(repository.findById(45553L)).thenReturn(Optional.empty());

        var result = service.ingest(List.of(rawDto("45553", LocalDateTime.now(), LocalDateTime.now())));

        assertThat(result.inserted()).isEqualTo(1);
        assertThat(result.updated()).isZero();
        assertThat(result.unchanged()).isZero();
        verify(repository).save(any(DefinitionOfSection.class));
    }

    @Test
    void updatesSectionWhenModifiedTimeMovedOn() {
        LocalDateTime oldTime = LocalDateTime.now().minusDays(1);
        DefinitionOfSection existing = mapper.toNewEntity(rawDto("1", oldTime, oldTime));
        when(repository.findById(1L)).thenReturn(Optional.of(existing));

        var result = service.ingest(List.of(rawDto("1", LocalDateTime.now(), oldTime)));

        assertThat(result.updated()).isEqualTo(1);
        assertThat(result.inserted()).isZero();
        assertThat(existing.getRecordUpdated()).isNotNull();
        verify(repository, times(1)).save(existing);
    }

    @Test
    void leavesUnchangedSectionAlone() {
        LocalDateTime time = LocalDateTime.now();
        DefinitionOfSectionRawDto dto = rawDto("1", time, time);
        DefinitionOfSection existing = mapper.toNewEntity(dto);
        when(repository.findById(1L)).thenReturn(Optional.of(existing));

        var result = service.ingest(List.of(dto));

        assertThat(result.unchanged()).isEqualTo(1);
        assertThat(result.inserted()).isZero();
        assertThat(result.updated()).isZero();
        verify(repository, never()).save(any());
    }

    private DefinitionOfSectionRawDto rawDto(String id, LocalDateTime modifiedTime, LocalDateTime geometryModifiedTime) {
        LineStringGeoJson geometry = new LineStringGeoJson("LineString", new double[][] { { 17.9, 58.9 }, { 17.91, 58.91 } });
        DefinitionOfSectionPropertiesDto properties = new DefinitionOfSectionPropertiesDto(
                "Test section", "VST", 1, modifiedTime, geometryModifiedTime, true, LocalDateTime.now(), 3, LocalDateTime.now());
        return new DefinitionOfSectionRawDto("Feature", id, geometry, properties);
    }
}
