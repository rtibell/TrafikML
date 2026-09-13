package com.tibell.trafficml.scheduler;

import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;
import com.tibell.trafficml.services.TrafficMessageClient;
import com.tibell.trafficml.services.TrafficMessageIngestionService;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Every-10-minutes (by default, see {@code trafficml.schedule.traffic-messages-cron})
 * pull of currently active traffic messages.
 */
@Component
@ConditionalOnProperty(prefix = "trafficml.schedule", name = "enabled", havingValue = "true", matchIfMissing = true)
public class TrafficMessageScheduler {

    private static final Logger log = LoggerFactory.getLogger(TrafficMessageScheduler.class);

    private final TrafficMessageClient client;
    private final TrafficMessageIngestionService ingestionService;

    public TrafficMessageScheduler(TrafficMessageClient client, TrafficMessageIngestionService ingestionService) {
        this.client = client;
        this.ingestionService = ingestionService;
    }

    @Scheduled(cron = "${trafficml.schedule.traffic-messages-cron}", zone = "${trafficml.schedule.zone}")
    public void fetchAndStore() {
        try {
            List<TrafficMessageRawDto> messages = client.fetchMessages();
            ingestionService.ingest(messages);
        } catch (Exception e) {
            log.error("TrafficMessage scheduled fetch failed", e);
        }
    }
}
