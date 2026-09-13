package com.tibell.trafficml.scheduler;

import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionRawDto;
import com.tibell.trafficml.services.DefinitionOfSectionClient;
import com.tibell.trafficml.services.DefinitionOfSectionIngestionService;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Once-a-day (06:00 by default, see {@code trafficml.schedule.definition-of-section-cron})
 * pull of the full section-definition set.
 */
@Component
@ConditionalOnProperty(prefix = "trafficml.schedule", name = "enabled", havingValue = "true", matchIfMissing = true)
public class DefinitionOfSectionScheduler {

    private static final Logger log = LoggerFactory.getLogger(DefinitionOfSectionScheduler.class);

    private final DefinitionOfSectionClient client;
    private final DefinitionOfSectionIngestionService ingestionService;

    public DefinitionOfSectionScheduler(DefinitionOfSectionClient client,
            DefinitionOfSectionIngestionService ingestionService) {
        this.client = client;
        this.ingestionService = ingestionService;
    }

    @Scheduled(cron = "${trafficml.schedule.definition-of-section-cron}", zone = "${trafficml.schedule.zone}")
    public void fetchAndStore() {
        try {
            List<DefinitionOfSectionRawDto> sections = client.fetchSections();
            ingestionService.ingest(sections);
        } catch (Exception e) {
            log.error("DefinitionOfSection scheduled fetch failed", e);
        }
    }
}
