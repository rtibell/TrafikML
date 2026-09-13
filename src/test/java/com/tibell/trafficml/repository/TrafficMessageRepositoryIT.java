package com.tibell.trafficml.repository;

import com.tibell.trafficml.entities.TrafficMessage;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.time.LocalDateTime;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

import tools.jackson.databind.ObjectMapper;

import jakarta.persistence.EntityManager;

import com.tibell.trafficml.mapper.GeoJsonConverter;
import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.model.PointGeoJson;
import com.tibell.trafficml.testsupport.AbstractIntegrationTest;

class TrafficMessageRepositoryIT extends AbstractIntegrationTest {

    @Autowired
    private TrafficMessageRepository repository;

    @Autowired
    private EntityManager entityManager;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void savesAndReloadsMessageIncludingGeometriesAndJsonColumn() {
        TrafficMessage message = new TrafficMessage(999004L);
        message.setTitle("IT test message");
        message.setMessage("Body");
        message.setWgs84Position(GeoJsonConverter.toPoint(
                new PointGeoJson("Point", new double[] { 18.0967, 59.5750 }), GeoJsonConverter.SRID_WGS84));
        message.setSweRef99Extent(GeoJsonConverter.toLineString(
                new LineStringGeoJson("LineString", new double[][] { { 675188.39, 6605920.66 }, { 675319.36, 6606228.81 } }),
                GeoJsonConverter.SRID_SWEREF99_TM));
        message.setScheduledOccurrences(objectMapper.valueToTree(java.util.List.of(java.util.Map.of("from", "08:00"))));
        message.setStartTime(LocalDateTime.now());
        message.setEndTime(LocalDateTime.now().plusDays(1));
        message.setRecordCreated(Instant.now());

        repository.saveAndFlush(message);
        entityManager.clear();

        TrafficMessage reloaded = repository.findById(999004L).orElseThrow();
        assertThat(reloaded.getTitle()).isEqualTo("IT test message");
        assertThat(reloaded.getWgs84Position().getX()).isEqualTo(18.0967);
        assertThat(reloaded.getWgs84Position().getSRID()).isEqualTo(4326);
        assertThat(reloaded.getSweRef99Extent().getSRID()).isEqualTo(3006);
        assertThat(reloaded.getScheduledOccurrences().get(0).get("from").asText()).isEqualTo("08:00");
    }

    @Test
    void existsByIdReflectsPersistedState() {
        assertThat(repository.existsById(999005L)).isFalse();

        TrafficMessage message = new TrafficMessage(999005L);
        message.setRecordCreated(Instant.now());
        repository.saveAndFlush(message);

        assertThat(repository.existsById(999005L)).isTrue();
    }
}
