package com.tibell.trafficml.scheduler;

import com.tibell.trafficml.model.speedofsection.SpeedOfSectionRawDto;
import com.tibell.trafficml.services.SpeedOfSectionClient;
import com.tibell.trafficml.services.SpeedOfSectionIngestionService;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Every-minute (07:00-19:00 by default, see
 * {@code trafficml.schedule.speed-of-section-cron}) pull of the current section speeds.
 */
@Component
@ConditionalOnProperty(prefix = "trafficml.schedule", name = "enabled", havingValue = "true", matchIfMissing = true)
public class SpeedOfSectionScheduler {

    private static final Logger log = LoggerFactory.getLogger(SpeedOfSectionScheduler.class);

    private final SpeedOfSectionClient client;
    private final SpeedOfSectionIngestionService ingestionService;

    public SpeedOfSectionScheduler(SpeedOfSectionClient client, SpeedOfSectionIngestionService ingestionService) {
        this.client = client;
        this.ingestionService = ingestionService;
    }

    @Scheduled(cron = "${trafficml.schedule.speed-of-section-cron}", zone = "${trafficml.schedule.zone}")
    public void fetchAndStore() {
        try {
            List<SpeedOfSectionRawDto> speeds = client.fetchSpeeds();
            ingestionService.ingest(speeds);
        } catch (Exception e) {
            log.error("SpeedOfSection scheduled fetch failed", e);
        }
    }
}
