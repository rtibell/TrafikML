package com.tibell.trafficml.mapper;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionPropertiesDto;
import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionRawDto;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDateTime;

import org.junit.jupiter.api.Test;

import com.tibell.trafficml.model.LineStringGeoJson;

class DefinitionOfSectionMapperTest {

    private final DefinitionOfSectionMapper mapper = new DefinitionOfSectionMapper();

    @Test
    void toIdConvertsStringIdToLong() {
        DefinitionOfSectionRawDto dto = rawDto("45553", LocalDateTime.now(), LocalDateTime.now(), LocalDateTime.now());

        assertThat(mapper.toId(dto)).isEqualTo(45553L);
    }

    @Test
    void toNewEntityCopiesAllFields() {
        LocalDateTime modifiedTime = LocalDateTime.parse("2026-05-21T08:30:52");
        LocalDateTime geometryModifiedTime = LocalDateTime.parse("2026-03-11T03:29:52");
        LocalDateTime activatedModifiedTime = LocalDateTime.parse("2026-03-11T03:29:48");
        DefinitionOfSectionRawDto dto = rawDto("45553", modifiedTime, geometryModifiedTime, activatedModifiedTime);

        DefinitionOfSection entity = mapper.toNewEntity(dto);

        assertThat(entity.getId()).isEqualTo(45553L);
        assertThat(entity.getName()).isEqualTo("Avfart Norrut trafikplats Osmo");
        assertThat(entity.getRegion()).isEqualTo("VST");
        assertThat(entity.getCountyNo()).isEqualTo(1);
        assertThat(entity.getModifiedTime()).isEqualTo(modifiedTime);
        assertThat(entity.getGeometryModifiedTime()).isEqualTo(geometryModifiedTime);
        assertThat(entity.getActivated()).isTrue();
        assertThat(entity.getActivatedModifiedTime()).isEqualTo(activatedModifiedTime);
        assertThat(entity.getAverageFunctionalRoadClass()).isEqualTo(3);
        assertThat(entity.getGeometry().getNumPoints()).isEqualTo(2);
    }

    @Test
    void hasChangedIsFalseWhenBothTimestampsAreUnchanged() {
        LocalDateTime modifiedTime = LocalDateTime.now();
        LocalDateTime geometryModifiedTime = LocalDateTime.now().minusDays(1);
        DefinitionOfSectionRawDto dto = rawDto("1", modifiedTime, geometryModifiedTime, LocalDateTime.now());
        DefinitionOfSection existing = mapper.toNewEntity(dto);

        assertThat(mapper.hasChanged(existing, dto)).isFalse();
    }

    @Test
    void hasChangedIsTrueWhenModifiedTimeMovedOn() {
        LocalDateTime original = LocalDateTime.now().minusDays(1);
        DefinitionOfSectionRawDto originalDto = rawDto("1", original, original, LocalDateTime.now());
        DefinitionOfSection existing = mapper.toNewEntity(originalDto);

        DefinitionOfSectionRawDto updatedDto = rawDto("1", LocalDateTime.now(), original, LocalDateTime.now());

        assertThat(mapper.hasChanged(existing, updatedDto)).isTrue();
    }

    @Test
    void hasChangedIsTrueWhenGeometryModifiedTimeMovedOn() {
        LocalDateTime modifiedTime = LocalDateTime.now();
        DefinitionOfSectionRawDto originalDto = rawDto("1", modifiedTime, modifiedTime.minusDays(1), LocalDateTime.now());
        DefinitionOfSection existing = mapper.toNewEntity(originalDto);

        DefinitionOfSectionRawDto updatedDto = rawDto("1", modifiedTime, LocalDateTime.now(), LocalDateTime.now());

        assertThat(mapper.hasChanged(existing, updatedDto)).isTrue();
    }

    private DefinitionOfSectionRawDto rawDto(String id, LocalDateTime modifiedTime, LocalDateTime geometryModifiedTime, LocalDateTime activatedModifiedTime) {
        LineStringGeoJson geometry = new LineStringGeoJson("LineString", new double[][] {
                { 17.910571, 58.972645 },
                { 17.911648, 58.974388 }
        });
        DefinitionOfSectionPropertiesDto properties = new DefinitionOfSectionPropertiesDto(
                "Avfart Norrut trafikplats Osmo", "VST", 1, modifiedTime, geometryModifiedTime, true, activatedModifiedTime,3,
                LocalDateTime.now());
        return new DefinitionOfSectionRawDto("Feature", id, geometry, properties);
    }
}
