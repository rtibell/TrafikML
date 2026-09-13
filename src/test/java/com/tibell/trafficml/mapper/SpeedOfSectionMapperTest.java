package com.tibell.trafficml.mapper;

import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.model.speedofsection.SpeedOfSectionRawDto;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDateTime;

import org.junit.jupiter.api.Test;

import com.tibell.trafficml.entities.DefinitionOfSection;

class SpeedOfSectionMapperTest {

    private final SpeedOfSectionMapper mapper = new SpeedOfSectionMapper();

    @Test
    void toSectionIdConvertsStringIdToLong() {
        SpeedOfSectionRawDto dto = new SpeedOfSectionRawDto("44469", "freeflow", 48, LocalDateTime.now());

        assertThat(mapper.toSectionId(dto)).isEqualTo(44469L);
    }

    @Test
    void toNewEntityBuildsCompositeKeyAndCopiesFields() {
        DefinitionOfSection section = new DefinitionOfSection(44469L);
        LocalDateTime measureTime = LocalDateTime.parse("2026-09-12T17:19:00");
        SpeedOfSectionRawDto dto = new SpeedOfSectionRawDto("44469", "freeflow", 48, measureTime);

        SpeedOfSection entity = mapper.toNewEntity(section, dto);

        assertThat(entity.getId().sectionId()).isEqualTo(44469L);
        assertThat(entity.getId().measureTime()).isEqualTo(measureTime);
        assertThat(entity.getStatus()).isEqualTo("freeflow");
        assertThat(entity.getSpeed()).isEqualTo(48);
        assertThat(entity.getSection()).isSameAs(section);
    }

    @Test
    void updateFieldsOverwritesStatusAndSpeed() {
        DefinitionOfSection section = new DefinitionOfSection(1L);
        SpeedOfSectionRawDto original = new SpeedOfSectionRawDto("1", "freeflow", 60, LocalDateTime.now());
        SpeedOfSection entity = mapper.toNewEntity(section, original);

        SpeedOfSectionRawDto updated = new SpeedOfSectionRawDto("1", "congested", 15, original.measureTime());
        mapper.updateFields(entity, updated);

        assertThat(entity.getStatus()).isEqualTo("congested");
        assertThat(entity.getSpeed()).isEqualTo(15);
    }
}
