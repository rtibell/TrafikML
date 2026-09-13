package com.tibell.trafficml.services;

import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.model.PointGeoJson;
import com.tibell.trafficml.testsupport.TestProperties;

@ExtendWith(MockitoExtension.class)
class TrafficMessageIngestionServiceTest {

    @Mock
    private TrafficMessagePersister persister;

    @Mock
    private MessageTemplateRenderer templateRenderer;

    @Mock
    private TrafficMessageFileStorageService fileStorageService;

    @Mock
    private SlackNotificationClient slackNotificationClient;

    private TrafficMessageIngestionService service;

    @BeforeEach
    void setUp() {
        service = new TrafficMessageIngestionService(persister, templateRenderer, fileStorageService,
                slackNotificationClient, TestProperties.withApiUrl("https://example.invalid"));
    }

    @Test
    void rendersStoresAndNotifiesOnlyForNewlyInsertedMessages() {
        TrafficMessageRawDto dto = dto(1L);
        when(persister.persistNewMessages(List.of(dto))).thenReturn(List.of(dto));
        when(templateRenderer.render(anyString(), any())).thenReturn("rendered text");

        var result = service.ingest(List.of(dto));

        assertThat(result.inserted()).isEqualTo(1);
        assertThat(result.alreadyKnown()).isZero();
        verify(fileStorageService).store(1L, "rendered text");
        verify(slackNotificationClient).send("rendered text");
    }

    @Test
    void skipsFileAndSlackForAlreadyKnownMessages() {
        TrafficMessageRawDto dto = dto(1L);
        when(persister.persistNewMessages(List.of(dto))).thenReturn(List.of());

        var result = service.ingest(List.of(dto));

        assertThat(result.inserted()).isZero();
        assertThat(result.alreadyKnown()).isEqualTo(1);
        verify(fileStorageService, never()).store(anyLong(), anyString());
        verify(slackNotificationClient, never()).send(anyString());
    }

    @Test
    void slackFailureDoesNotPreventProcessingOtherMessages() {
        TrafficMessageRawDto first = dto(1L);
        TrafficMessageRawDto second = dto(2L);
        when(persister.persistNewMessages(List.of(first, second))).thenReturn(List.of(first, second));
        when(templateRenderer.render(anyString(), any())).thenReturn("rendered text");
        doThrow(new RuntimeException("Slack down")).when(slackNotificationClient).send(anyString());

        var result = service.ingest(List.of(first, second));

        assertThat(result.inserted()).isEqualTo(2);
        verify(fileStorageService, times(2)).store(anyLong(), anyString());
    }

    private TrafficMessageRawDto dto(Long id) {
        return new TrafficMessageRawDto(
                id, "VST", "Title " + id, "Message", "provider", "type", "code", "traffic",
                new PointGeoJson("Point", new double[] { 18.0, 59.0 }),
                new LineStringGeoJson("LineString", new double[][] { { 1, 2 }, { 3, 4 } }),
                "Both", LocalDateTime.now(), LocalDateTime.now(), List.of(), LocalDateTime.now(), 1, 1, true, false,
                "https://example.invalid", 1, "text", 0);
    }
}
