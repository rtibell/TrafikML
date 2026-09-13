package com.tibell.trafficml.configuration;

import jakarta.validation.constraints.NotBlank;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;
import org.springframework.validation.annotation.Validated;

/**
 * Root binding for the {@code trafficml.*} configuration tree (see application.yml).
 * All environment-specific and dynamic values (URLs, cron expressions, Slack settings,
 * message templates, storage locations) are kept here rather than hard-coded.
 */
@Validated
@ConfigurationProperties(prefix = "trafficml")
public record TrafficMlProperties(
        @NestedConfigurationProperty Api api,
        @NestedConfigurationProperty Schedule schedule,
        @NestedConfigurationProperty Slack slack,
        @NestedConfigurationProperty TrafficMessage trafficMessage) {

    public record Api(
            @NotBlank String speedOfSectionUrl,
            @NotBlank String definitionOfSectionUrl,
            @NotBlank String trafficMessagesUrl) {
    }

    public record Schedule(
            boolean enabled,
            @NotBlank String zone,
            @NotBlank String speedOfSectionCron,
            @NotBlank String definitionOfSectionCron,
            @NotBlank String trafficMessagesCron) {
    }

    public record Slack(
            String webhookUrl,
            @NotBlank String channel,
            boolean enabled) {
    }

    public record TrafficMessage(
            @NotBlank String template,
            @NotBlank String storageDir) {
    }
}
