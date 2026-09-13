package com.tibell.trafficml.mapper;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionPropertiesDto;
import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionRawDto;

import java.util.Objects;

import org.springframework.stereotype.Component;

import com.tibell.trafficml.mapper.GeoJsonConverter;

/**
 * Maps the raw upstream DTO onto {@link DefinitionOfSection}, and decides whether an
 * already-persisted section needs updating.
 */
@Component
public class DefinitionOfSectionMapper {

    public Long toId(DefinitionOfSectionRawDto dto) {
        return Long.valueOf(dto.id());
    }

    public DefinitionOfSection toNewEntity(DefinitionOfSectionRawDto dto) {
        DefinitionOfSection entity = new DefinitionOfSection(toId(dto));
        applyFields(entity, dto);
        return entity;
    }

    public void updateFields(DefinitionOfSection entity, DefinitionOfSectionRawDto dto) {
        applyFields(entity, dto);
    }

    /**
     * The upstream record is considered "updated" when either timestamp it exposes for
     * change tracking has moved on, per the {@code id}/{@code modifiedTime}/
     * {@code geometryModifiedTime} composite key described in the spec.
     */
    public boolean hasChanged(DefinitionOfSection existing, DefinitionOfSectionRawDto dto) {
        DefinitionOfSectionPropertiesDto properties = dto.properties();
        return !Objects.equals(existing.getModifiedTime(), properties.modifiedTime())
                || !Objects.equals(existing.getGeometryModifiedTime(), properties.geometryModifiedTime());
    }

    private void applyFields(DefinitionOfSection entity, DefinitionOfSectionRawDto dto) {
        DefinitionOfSectionPropertiesDto properties = dto.properties();
        entity.setName(properties.name());
        entity.setRegion(properties.region());
        entity.setCountyNo(properties.countyNo());
        entity.setModifiedTime(properties.modifiedTime());
        entity.setGeometryModifiedTime(properties.geometryModifiedTime());
        entity.setActivated(properties.activated());
        entity.setActivatedModifiedTime(properties.activatedModifiedTime());
        entity.setAverageFunctionalRoadClass(properties.averageFunctionalRoadClass());
        entity.setImportedTime(properties.importedTime());
        entity.setGeometry(GeoJsonConverter.toLineString(dto.geometry(), GeoJsonConverter.SRID_WGS84));
    }
}
