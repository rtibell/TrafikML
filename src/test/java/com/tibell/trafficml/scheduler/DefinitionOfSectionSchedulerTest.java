package com.tibell.trafficml.scheduler;

import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionRawDto;
import com.tibell.trafficml.services.DefinitionOfSectionClient;
import com.tibell.trafficml.services.DefinitionOfSectionIngestionService;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class DefinitionOfSectionSchedulerTest {

    @Mock
    private DefinitionOfSectionClient client;

    @Mock
    private DefinitionOfSectionIngestionService ingestionService;

    @Test
    void fetchAndStoreDelegatesToClientAndIngestionService() {
        List<DefinitionOfSectionRawDto> sections = List.of();
        when(client.fetchSections()).thenReturn(sections);

        new DefinitionOfSectionScheduler(client, ingestionService).fetchAndStore();

        verify(client).fetchSections();
        verify(ingestionService).ingest(sections);
    }

    @Test
    void fetchAndStoreSwallowsExceptionsSoOneBadRunDoesNotKillTheScheduler() {
        when(client.fetchSections()).thenThrow(new RuntimeException("upstream unavailable"));

        assertThatCode(() -> new DefinitionOfSectionScheduler(client, ingestionService).fetchAndStore())
                .doesNotThrowAnyException();
    }
}
