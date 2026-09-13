package com.tibell.trafficml.services;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.mapper.DefinitionOfSectionMapper;
import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionRawDto;
import com.tibell.trafficml.repository.DefinitionOfSectionRepository;

import java.time.Instant;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Applies fetched {@link DefinitionOfSectionRawDto} records to the database: insert if
 * the section id is unseen, update if {@code modifiedTime}/{@code geometryModifiedTime}
 * moved on, otherwise leave the row untouched.
 */
@Service
public class DefinitionOfSectionIngestionService {

    private static final Logger log = LoggerFactory.getLogger(DefinitionOfSectionIngestionService.class);

    private final DefinitionOfSectionRepository repository;
    private final DefinitionOfSectionMapper mapper;

    public DefinitionOfSectionIngestionService(DefinitionOfSectionRepository repository,
            DefinitionOfSectionMapper mapper) {
        this.repository = repository;
        this.mapper = mapper;
    }

    @Transactional
    public IngestionResult ingest(List<DefinitionOfSectionRawDto> dtos) {
        int inserted = 0;
        int updated = 0;
        int unchanged = 0;
        Instant now = Instant.now();

        for (DefinitionOfSectionRawDto dto : dtos) {
            Long id = mapper.toId(dto);
            var existing = repository.findById(id);
            if (existing.isEmpty()) {
                DefinitionOfSection entity = mapper.toNewEntity(dto);
                entity.setRecordCreated(now);
                repository.save(entity);
                inserted++;
            } else if (mapper.hasChanged(existing.get(), dto)) {
                DefinitionOfSection entity = existing.get();
                mapper.updateFields(entity, dto);
                entity.setRecordUpdated(now);
                repository.save(entity);
                updated++;
            } else {
                unchanged++;
            }
        }

        log.info("DefinitionOfSection ingestion complete: {} inserted, {} updated, {} unchanged (of {} received)",
                inserted, updated, unchanged, dtos.size());
        return new IngestionResult(inserted, updated, unchanged);
    }

    public record IngestionResult(int inserted, int updated, int unchanged) {
    }
}
