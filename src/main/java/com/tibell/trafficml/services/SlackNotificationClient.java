package com.tibell.trafficml.services;

import java.net.URI;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import com.tibell.trafficml.configuration.TrafficMlProperties;

/**
 * Posts a rendered traffic message to Slack via an incoming webhook. Silently becomes a
 * no-op when {@code trafficml.slack.enabled=false} or no webhook URL is configured, so
 * dev/test environments don't need a real Slack workspace.
 */
@Component
public class SlackNotificationClient {

    private static final Logger log = LoggerFactory.getLogger(SlackNotificationClient.class);

    private final RestClient restClient;
    private final TrafficMlProperties.Slack slackProperties;

    public SlackNotificationClient(RestClient.Builder trafficMlRestClientBuilder, TrafficMlProperties properties) {
        this.restClient = trafficMlRestClientBuilder.build();
        this.slackProperties = properties.slack();
    }

    public void send(String text) {
        if (!slackProperties.enabled() || slackProperties.webhookUrl() == null || slackProperties.webhookUrl().isBlank()) {
            log.debug("Slack notifications disabled or webhook URL not configured; skipping send");
            return;
        }

        SlackMessage payload = new SlackMessage(slackProperties.channel(), text);
        restClient.post()
                .uri(URI.create(slackProperties.webhookUrl()))
                .body(payload)
                .retrieve()
                .toBodilessEntity();
    }

    private record SlackMessage(String channel, String text) {
    }
}
