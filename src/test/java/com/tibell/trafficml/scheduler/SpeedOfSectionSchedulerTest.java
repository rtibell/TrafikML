package com.tibell.trafficml.scheduler;

import com.tibell.trafficml.model.speedofsection.SpeedOfSectionRawDto;
import com.tibell.trafficml.services.SpeedOfSectionClient;
import com.tibell.trafficml.services.SpeedOfSectionIngestionService;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class SpeedOfSectionSchedulerTest {

    @Mock
    private SpeedOfSectionClient client;

    @Mock
    private SpeedOfSectionIngestionService ingestionService;

    @Test
    void fetchAndStoreDelegatesToClientAndIngestionService() {
        List<SpeedOfSectionRawDto> speeds = List.of();
        when(client.fetchSpeeds()).thenReturn(speeds);

        new SpeedOfSectionScheduler(client, ingestionService).fetchAndStore();

        verify(client).fetchSpeeds();
        verify(ingestionService).ingest(speeds);
    }

    @Test
    void fetchAndStoreSwallowsExceptionsSoOneBadRunDoesNotKillTheScheduler() {
        when(client.fetchSpeeds()).thenThrow(new RuntimeException("upstream unavailable"));

        assertThatCode(() -> new SpeedOfSectionScheduler(client, ingestionService).fetchAndStore())
                .doesNotThrowAnyException();
    }
}
