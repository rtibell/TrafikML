package com.tibell.trafficml.services;

import com.tibell.trafficml.entities.TrafficMessage;
import com.tibell.trafficml.mapper.TrafficMessageMapper;
import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;
import com.tibell.trafficml.repository.TrafficMessageRepository;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Transactional insert-if-unseen persistence for traffic messages, kept as its own bean
 * (rather than a method on {@link TrafficMessageIngestionService}) so the
 * {@code @Transactional} boundary is a real proxy call and not a same-class
 * self-invocation, which Spring would silently ignore.
 */
@Service
class TrafficMessagePersister {

    private final TrafficMessageRepository repository;
    private final TrafficMessageMapper mapper;

    TrafficMessagePersister(TrafficMessageRepository repository, TrafficMessageMapper mapper) {
        this.repository = repository;
        this.mapper = mapper;
    }

    @Transactional
    List<TrafficMessageRawDto> persistNewMessages(List<TrafficMessageRawDto> dtos) {
        Instant now = Instant.now();
        List<TrafficMessageRawDto> newlyInserted = new ArrayList<>();

        for (TrafficMessageRawDto dto : dtos) {
            if (repository.existsById(dto.id())) {
                continue;
            }
            TrafficMessage entity = mapper.toEntity(dto);
            entity.setRecordCreated(now);
            repository.save(entity);
            newlyInserted.add(dto);
        }
        return newlyInserted;
    }
}
