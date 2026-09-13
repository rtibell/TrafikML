package com.tibell.trafficml.mapper;

import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.model.speedofsection.SpeedOfSectionRawDto;

import org.springframework.stereotype.Component;

import com.tibell.trafficml.entities.DefinitionOfSection;

@Component
public class SpeedOfSectionMapper {

    public Long toSectionId(SpeedOfSectionRawDto dto) {
        return Long.valueOf(dto.id());
    }

    public SpeedOfSection toNewEntity(DefinitionOfSection section, SpeedOfSectionRawDto dto) {
        SpeedOfSection entity = new SpeedOfSection(section, dto.measureTime());
        applyFields(entity, dto);
        return entity;
    }

    public void updateFields(SpeedOfSection entity, SpeedOfSectionRawDto dto) {
        applyFields(entity, dto);
    }

    private void applyFields(SpeedOfSection entity, SpeedOfSectionRawDto dto) {
        entity.setStatus(dto.status());
        entity.setSpeed(dto.speed());
    }
}
