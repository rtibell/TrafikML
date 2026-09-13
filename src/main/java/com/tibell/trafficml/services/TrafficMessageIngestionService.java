package com.tibell.trafficml.services;

import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.tibell.trafficml.configuration.TrafficMlProperties;

/**
 * Persists newly-seen traffic messages (by id) and, for each one actually inserted,
 * renders the configured template, writes it to disk, and forwards it to Slack.
 */
@Service
public class TrafficMessageIngestionService {

    private static final Logger log = LoggerFactory.getLogger(TrafficMessageIngestionService.class);

    private final TrafficMessagePersister persister;
    private final MessageTemplateRenderer templateRenderer;
    private final TrafficMessageFileStorageService fileStorageService;
    private final SlackNotificationClient slackNotificationClient;
    private final String template;

    public TrafficMessageIngestionService(TrafficMessagePersister persister, MessageTemplateRenderer templateRenderer,
            TrafficMessageFileStorageService fileStorageService, SlackNotificationClient slackNotificationClient,
            TrafficMlProperties properties) {
        this.persister = persister;
        this.templateRenderer = templateRenderer;
        this.fileStorageService = fileStorageService;
        this.slackNotificationClient = slackNotificationClient;
        this.template = properties.trafficMessage().template();
    }

    public IngestionResult ingest(List<TrafficMessageRawDto> dtos) {
        List<TrafficMessageRawDto> newlyInserted = persister.persistNewMessages(dtos);

        // File/Slack side effects run after the DB transaction has committed, so a slow
        // or failing webhook call never holds a database connection open nor rolls back
        // an otherwise-successful insert.
        for (TrafficMessageRawDto dto : newlyInserted) {
            String renderedMessage = templateRenderer.render(template, dto);
            fileStorageService.store(dto.id(), renderedMessage);
            notifySlack(dto.id(), renderedMessage);
        }

        int alreadyKnown = dtos.size() - newlyInserted.size();
        log.info("TrafficMessage ingestion complete: {} inserted, {} already known (of {} received)",
                newlyInserted.size(), alreadyKnown, dtos.size());
        return new IngestionResult(newlyInserted.size(), alreadyKnown);
    }

    private void notifySlack(Long id, String renderedMessage) {
        try {
            slackNotificationClient.send(renderedMessage);
        } catch (Exception e) {
            // A Slack outage must not roll back the DB insert or stop the file being written.
            log.error("Failed to send Slack notification for traffic message {}", id, e);
        }
    }

    public record IngestionResult(int inserted, int alreadyKnown) {
    }
}
