package com.tibell.trafficml.controller;

import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.repository.SpeedOfSectionRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.data.domain.Pageable;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.assertj.MockMvcTester;

import com.tibell.trafficml.entities.DefinitionOfSection;

@WebMvcTest(SpeedOfSectionController.class)
class SpeedOfSectionControllerTest {

    @Autowired
    private MockMvcTester mvc;

    @MockitoBean
    private SpeedOfSectionRepository repository;

    @Test
    void historyReturnsMeasurementsForSection() {
        DefinitionOfSection section = new DefinitionOfSection(44469L);
        SpeedOfSection measurement = new SpeedOfSection(section, LocalDateTime.parse("2026-09-12T17:19:00"));
        measurement.setStatus("freeflow");
        measurement.setSpeed(48);
        when(repository.findByIdSectionIdOrderByIdMeasureTimeDesc(eq(44469L), any(Pageable.class)))
                .thenReturn(List.of(measurement));

        assertThat(mvc.get().uri("/api/v1/sections/44469/speed"))
                .hasStatusOk()
                .bodyJson()
                .extractingPath("$[0].speed").isEqualTo(48);
    }
}
