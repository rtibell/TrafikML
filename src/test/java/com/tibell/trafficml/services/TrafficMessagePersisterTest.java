package com.tibell.trafficml.services;

import com.tibell.trafficml.entities.TrafficMessage;
import com.tibell.trafficml.mapper.TrafficMessageMapper;
import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;
import com.tibell.trafficml.repository.TrafficMessageRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import tools.jackson.databind.ObjectMapper;

import com.tibell.trafficml.model.LineStringGeoJson;
import com.tibell.trafficml.model.PointGeoJson;

@ExtendWith(MockitoExtension.class)
class TrafficMessagePersisterTest {

    @Mock
    private TrafficMessageRepository repository;

    private TrafficMessagePersister persister;

    @BeforeEach
    void setUp() {
        // Built here rather than as a field initializer: field initializers run before
        // MockitoExtension injects @Mock fields, so `repository` would still be null.
        persister = new TrafficMessagePersister(repository, new TrafficMessageMapper(new ObjectMapper()));
    }

    @Test
    void persistsOnlyMessagesNotAlreadyKnown() {
        TrafficMessageRawDto known = dto(1L);
        TrafficMessageRawDto unseen = dto(2L);
        when(repository.existsById(1L)).thenReturn(true);
        when(repository.existsById(2L)).thenReturn(false);

        List<TrafficMessageRawDto> newlyInserted = persister.persistNewMessages(List.of(known, unseen));

        assertThat(newlyInserted).containsExactly(unseen);
        verify(repository).save(any(TrafficMessage.class));
    }

    @Test
    void doesNothingWhenAllMessagesAreAlreadyKnown() {
        when(repository.existsById(1L)).thenReturn(true);

        List<TrafficMessageRawDto> newlyInserted = persister.persistNewMessages(List.of(dto(1L)));

        assertThat(newlyInserted).isEmpty();
        verify(repository, never()).save(any());
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
