package com.tibell.trafficml.controller;

import com.tibell.trafficml.model.speedofsection.MachineLearningSpeedOfSectionView;
import com.tibell.trafficml.services.SpeedOfSectionMLData;

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

@WebMvcTest(SpeedOfSectionMLDataController.class)
class SpeedOfSectionMLDataControllerTest {

    @Autowired
    private MockMvcTester mvc;

    @MockitoBean
    private SpeedOfSectionMLData mlData;

    @Test
    void mlSpeedDataReturnsFeatureRowsForSection() {
        MachineLearningSpeedOfSectionView view = new MachineLearningSpeedOfSectionView(
                44469L, LocalDateTime.parse("2026-09-13T13:44:00"), "freeflow", 0, 70, 6, 48, 0, 464, 9, 14);
        when(mlData.forSection(eq(44469L), any(Pageable.class))).thenReturn(List.of(view));

        var result = mvc.get().uri("/api/v1/sections/44469/ml-speed-data").exchange();

        assertThat(result).hasStatusOk();
        assertThat(result).bodyJson().extractingPath("$[0].statusEnum").isEqualTo(0);
        assertThat(result).bodyJson().extractingPath("$[0].speed").isEqualTo(70);
        assertThat(result).bodyJson().extractingPath("$[0].holidayNum").isEqualTo(14);
    }
}
