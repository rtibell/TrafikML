package com.tibell.trafficml.services;

import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.entities.SpeedOfSectionId;
import com.tibell.trafficml.mapper.SpeedOfSectionMapper;
import com.tibell.trafficml.model.speedofsection.SpeedOfSectionRawDto;
import com.tibell.trafficml.repository.SpeedOfSectionRepository;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.repository.DefinitionOfSectionRepository;

/**
 * Applies fetched {@link SpeedOfSectionRawDto} records to the database, using
 * {@code (id, measureTime)} as the composite key to tell new measurements from ones
 * that already exist.
 */
@Service
public class SpeedOfSectionIngestionService {

    private static final Logger log = LoggerFactory.getLogger(SpeedOfSectionIngestionService.class);

    private final SpeedOfSectionRepository repository;
    private final DefinitionOfSectionRepository sectionRepository;
    private final SpeedOfSectionMapper mapper;

    public SpeedOfSectionIngestionService(SpeedOfSectionRepository repository,
            DefinitionOfSectionRepository sectionRepository, SpeedOfSectionMapper mapper) {
        this.repository = repository;
        this.sectionRepository = sectionRepository;
        this.mapper = mapper;
    }

    @Transactional
    public IngestionResult ingest(List<SpeedOfSectionRawDto> dtos) {
        int inserted = 0;
        int updated = 0;
        int skippedUnknownSection = 0;
        Instant now = Instant.now();

        for (SpeedOfSectionRawDto dto : dtos) {
            Long sectionId = mapper.toSectionId(dto);
            SpeedOfSectionId id = new SpeedOfSectionId(sectionId, dto.measureTime());
            Optional<SpeedOfSection> existing = repository.findById(id);

            if (existing.isPresent()) {
                SpeedOfSection entity = existing.get();
                mapper.updateFields(entity, dto);
                entity.setRecordUpdated(now);
                repository.save(entity);
                updated++;
                continue;
            }

            Optional<DefinitionOfSection> section = sectionRepository.findById(sectionId);
            if (section.isEmpty()) {
                // Section definitions are refreshed once a day; a speed reading for a
                // section we don't know yet is skipped rather than failing the whole batch.
                log.warn("Skipping speed measurement for unknown section id {} at {}", sectionId, dto.measureTime());
                skippedUnknownSection++;
                continue;
            }

            SpeedOfSection entity = mapper.toNewEntity(section.get(), dto);
            entity.setRecordCreated(now);
            repository.save(entity);
            inserted++;
        }

        log.info("SpeedOfSection ingestion complete: {} inserted, {} updated, {} skipped (of {} received)",
                inserted, updated, skippedUnknownSection, dtos.size());
        return new IngestionResult(inserted, updated, skippedUnknownSection);
    }

    public record IngestionResult(int inserted, int updated, int skippedUnknownSection) {
    }
}
