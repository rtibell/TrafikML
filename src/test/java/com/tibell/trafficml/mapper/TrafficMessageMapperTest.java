package com.tibell.trafficml.mapper;

import com.tibell.trafficml.entities.TrafficMessage;
import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import tools.jackson.databind.ObjectMapper;

import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.model.PointGeoJson;

class TrafficMessageMapperTest {

    private final TrafficMessageMapper mapper = new TrafficMessageMapper(new ObjectMapper());

    @Test
    void toEntityCopiesScalarAndGeometryFields() {
        TrafficMessageRawDto dto = rawDto();

        TrafficMessage entity = mapper.toEntity(dto);

        assertThat(entity.getId()).isEqualTo(11851210L);
        assertThat(entity.getTitle()).isEqualTo("Vag 950 vid Granby");
        assertThat(entity.getWgs84Position().getX()).isEqualTo(18.0967);
        assertThat(entity.getWgs84Position().getSRID()).isEqualTo(4326);
        assertThat(entity.getSweRef99Extent().getNumPoints()).isEqualTo(2);
        assertThat(entity.getSweRef99Extent().getSRID()).isEqualTo(3006);
        assertThat(entity.getSeverity()).isEqualTo(2);
        assertThat(entity.getIsFuture()).isTrue();
    }

    @Test
    void toEntitySerializesScheduledOccurrencesAsJson() {
        TrafficMessageRawDto dto = rawDtoWithOccurrence();

        TrafficMessage entity = mapper.toEntity(dto);

        assertThat(entity.getScheduledOccurrences().isArray()).isTrue();
        assertThat(entity.getScheduledOccurrences().get(0).get("from").asText()).isEqualTo("08:00");
    }

    private TrafficMessageRawDto rawDto() {
        return new TrafficMessageRawDto(
                11851210L, "VST", "Vag 950 vid Granby", "<span>Message</span>", "Trafik Stockholm", "Vagarbete",
                "Vagarbete", "Vagtrafik",
                new PointGeoJson("Point", new double[] { 18.0967, 59.5750 }),
                new LineStringGeoJson("LineString", new double[][] { { 675188.39, 6605920.66 }, { 675319.36, 6606228.81 } }),
                "Both", LocalDateTime.now(), LocalDateTime.now().plusDays(1), List.of(),
                LocalDateTime.now(), 162, 2, true, false, "https://trafiken.nu/details", 4347, "Liten paverkan", 0);
    }

    private TrafficMessageRawDto rawDtoWithOccurrence() {
        TrafficMessageRawDto base = rawDto();
        return new TrafficMessageRawDto(
                base.id(), base.region(), base.title(), base.message(), base.provider(), base.messageType(),
                base.messageCode(), base.trafficType(), base.wgs84Position(), base.sweRef99Extent(),
                base.affectedDirection(), base.startTime(), base.endTime(),
                List.of(Map.of("from", "08:00", "to", "09:00")), base.versionTime(), base.iconId(), base.severity(),
                base.isFuture(), base.roadClosed(), base.detailsPath(), base.sortIndex(), base.severityText(),
                base.workState());
    }
}
