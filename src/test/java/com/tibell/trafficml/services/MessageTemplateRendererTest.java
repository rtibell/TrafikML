package com.tibell.trafficml.services;

import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.Test;

import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.model.PointGeoJson;

class MessageTemplateRendererTest {

    private final MessageTemplateRenderer renderer = new MessageTemplateRenderer();

    @Test
    void replacesKnownPlaceholders() {
        String template = "{{title}}: {{message}} ({{severityText}})";

        String rendered = renderer.render(template, dto("Title", "<b>Body</b>", "Liten paverkan"));

        assertThat(rendered).isEqualTo("Title: Body (Liten paverkan)");
    }

    @Test
    void stripsHtmlTagsFromMessage() {
        String rendered = renderer.render("{{message}}", dto("t", "<span>Akut</span><br/>arbete", "s"));

        assertThat(rendered).isEqualTo("Akutarbete");
    }

    @Test
    void unknownPlaceholderIsReplacedWithEmptyString() {
        String rendered = renderer.render("[{{doesNotExist}}]", dto("t", "m", "s"));

        assertThat(rendered).isEqualTo("[]");
    }

    private TrafficMessageRawDto dto(String title, String message, String severityText) {
        return new TrafficMessageRawDto(
                1L, "VST", title, message, "provider", "type", "code", "traffic",
                new PointGeoJson("Point", new double[] { 18.0, 59.0 }),
                new LineStringGeoJson("LineString", new double[][] { { 1, 2 } }),
                "Both", LocalDateTime.now(), LocalDateTime.now(), List.of(), LocalDateTime.now(), 1, 1, true, false,
                "https://example.invalid", 1, severityText, 0);
    }
}
