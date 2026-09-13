package com.tibell.trafficml.mapper;

import com.tibell.trafficml.entities.TrafficMessage;
import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;

import tools.jackson.databind.ObjectMapper;

import org.springframework.stereotype.Component;

import com.tibell.trafficml.mapper.GeoJsonConverter;

@Component
public class TrafficMessageMapper {

    private final ObjectMapper objectMapper;

    public TrafficMessageMapper(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public TrafficMessage toEntity(TrafficMessageRawDto dto) {
        TrafficMessage entity = new TrafficMessage(dto.id());
        entity.setRegion(dto.region());
        entity.setTitle(dto.title());
        entity.setMessage(dto.message());
        entity.setProvider(dto.provider());
        entity.setMessageType(dto.messageType());
        entity.setMessageCode(dto.messageCode());
        entity.setTrafficType(dto.trafficType());
        entity.setWgs84Position(GeoJsonConverter.toPoint(dto.wgs84Position(), GeoJsonConverter.SRID_WGS84));
        entity.setSweRef99Extent(
                GeoJsonConverter.toLineString(dto.sweRef99Extent(), GeoJsonConverter.SRID_SWEREF99_TM));
        entity.setAffectedDirection(dto.affectedDirection());
        entity.setStartTime(dto.startTime());
        entity.setEndTime(dto.endTime());
        entity.setScheduledOccurrences(objectMapper.valueToTree(dto.scheduledOccurrences()));
        entity.setVersionTime(dto.versionTime());
        entity.setIconId(dto.iconId());
        entity.setSeverity(dto.severity());
        entity.setIsFuture(dto.isFuture());
        entity.setRoadClosed(dto.roadClosed());
        entity.setDetailsPath(dto.detailsPath());
        entity.setSortIndex(dto.sortIndex());
        entity.setSeverityText(dto.severityText());
        entity.setWorkState(dto.workState());
        return entity;
    }
}
