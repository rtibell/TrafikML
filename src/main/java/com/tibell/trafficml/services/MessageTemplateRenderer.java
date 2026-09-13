package com.tibell.trafficml.services;

import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;

import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.springframework.stereotype.Component;

/**
 * Renders the {@code trafficml.traffic-message.template} configured in application.yml,
 * replacing {@code {{placeholder}}} tokens with values from a traffic message.
 * A handful of named placeholders (rather than positional {@code %s} arguments) keeps
 * the template self-documenting for whoever edits application.yml.
 */
@Component
public class MessageTemplateRenderer {

    private static final Pattern PLACEHOLDER = Pattern.compile("\\{\\{(\\w+)}}");

    public String render(String template, TrafficMessageRawDto dto) {
        Map<String, String> values = Map.of(
                "id", String.valueOf(dto.id()),
                "title", nullToEmpty(dto.title()),
                "message", stripHtml(nullToEmpty(dto.message())),
                "messageType", nullToEmpty(dto.messageType()),
                "trafficType", nullToEmpty(dto.trafficType()),
                "severityText", nullToEmpty(dto.severityText()),
                "startTime", String.valueOf(dto.startTime()),
                "endTime", String.valueOf(dto.endTime()),
                "detailsPath", nullToEmpty(dto.detailsPath()));

        Matcher matcher = PLACEHOLDER.matcher(template);
        StringBuilder result = new StringBuilder();
        while (matcher.find()) {
            String value = values.getOrDefault(matcher.group(1), "");
            matcher.appendReplacement(result, Matcher.quoteReplacement(value));
        }
        matcher.appendTail(result);
        return result.toString();
    }

    private String stripHtml(String value) {
        return value.replaceAll("<[^>]*>", "");
    }

    private String nullToEmpty(String value) {
        return value == null ? "" : value;
    }
}
