package com.tibell.trafficml.repository;

import com.tibell.trafficml.entities.DefinitionOfSection;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDateTime;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

import jakarta.persistence.EntityManager;

import com.tibell.trafficml.mapper.GeoJsonConverter;
import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.testsupport.AbstractIntegrationTest;

class DefinitionOfSectionRepositoryIT extends AbstractIntegrationTest {

    @Autowired
    private DefinitionOfSectionRepository repository;

    @Autowired
    private EntityManager entityManager;

    @Test
    void savesAndReloadsSectionIncludingGeometry() {
        DefinitionOfSection section = new DefinitionOfSection(999001L);
        section.setName("IT test section");
        section.setRegion("VST");
        section.setCountyNo(1);
        section.setModifiedTime(LocalDateTime.parse("2026-05-21T08:30:52"));
        section.setGeometryModifiedTime(LocalDateTime.parse("2026-03-11T03:29:52"));
        section.setActivated(true);
        section.setActivatedModifiedTime(LocalDateTime.parse("2026-03-11T03:29:48"));
        section.setAverageFunctionalRoadClass(3);
        section.setRecordCreated(java.time.Instant.now());
        section.setGeometry(GeoJsonConverter.toLineString(
                new LineStringGeoJson("LineString", new double[][] { { 17.910571, 58.972645 }, { 17.914212, 58.977026 } }),
                GeoJsonConverter.SRID_WGS84));

        repository.saveAndFlush(section);
        entityManager.clear();

        DefinitionOfSection reloaded = repository.findById(999001L).orElseThrow();
        assertThat(reloaded.getName()).isEqualTo("IT test section");
        assertThat(reloaded.getGeometry().getNumPoints()).isEqualTo(2);
        assertThat(reloaded.getGeometry().getCoordinateN(0).x).isEqualTo(17.910571);
        assertThat(reloaded.getRecordCreated()).isNotNull();
    }
}
