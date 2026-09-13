package com.tibell.trafficml.scheduler;

import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;
import com.tibell.trafficml.services.TrafficMessageClient;
import com.tibell.trafficml.services.TrafficMessageIngestionService;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class TrafficMessageSchedulerTest {

    @Mock
    private TrafficMessageClient client;

    @Mock
    private TrafficMessageIngestionService ingestionService;

    @Test
    void fetchAndStoreDelegatesToClientAndIngestionService() {
        List<TrafficMessageRawDto> messages = List.of();
        when(client.fetchMessages()).thenReturn(messages);

        new TrafficMessageScheduler(client, ingestionService).fetchAndStore();

        verify(client).fetchMessages();
        verify(ingestionService).ingest(messages);
    }

    @Test
    void fetchAndStoreSwallowsExceptionsSoOneBadRunDoesNotKillTheScheduler() {
        when(client.fetchMessages()).thenThrow(new RuntimeException("upstream unavailable"));

        assertThatCode(() -> new TrafficMessageScheduler(client, ingestionService).fetchAndStore())
                .doesNotThrowAnyException();
    }
}
