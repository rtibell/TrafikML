package com.tibell.trafficml.controller;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.repository.DefinitionOfSectionRepository;

import static org.mockito.Mockito.when;

import java.time.LocalDateTime;
import java.util.Optional;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.assertj.MockMvcTester;

@WebMvcTest(DefinitionOfSectionController.class)
class DefinitionOfSectionControllerTest {

    @Autowired
    private MockMvcTester mvc;

    @MockitoBean
    private DefinitionOfSectionRepository repository;

    @Test
    void getReturnsSectionAsJson() {
        DefinitionOfSection section = new DefinitionOfSection(45553L);
        section.setName("Test section");
        section.setModifiedTime(LocalDateTime.parse("2026-05-21T08:30:52"));
        section.setGeometryModifiedTime(LocalDateTime.parse("2026-03-11T03:29:52"));
        when(repository.findById(45553L)).thenReturn(Optional.of(section));

        Assertions.assertThat(mvc.get().uri("/api/v1/sections/45553"))
                .hasStatusOk()
                .bodyJson()
                .extractingPath("$.id").isEqualTo(45553);
    }

    @Test
    void getReturns404WhenSectionUnknown() {
        when(repository.findById(1L)).thenReturn(Optional.empty());

        Assertions.assertThat(mvc.get().uri("/api/v1/sections/1")).hasStatus4xxClientError();
    }

    @Test
    void listReturnsPageContent() {
        DefinitionOfSection section = new DefinitionOfSection(1L);
        section.setModifiedTime(LocalDateTime.now());
        section.setGeometryModifiedTime(LocalDateTime.now());
        Page<DefinitionOfSection> page = new PageImpl<>(java.util.List.of(section));
        when(repository.findAll(org.mockito.ArgumentMatchers.any(org.springframework.data.domain.Pageable.class)))
                .thenReturn(page);

        Assertions.assertThat(mvc.get().uri("/api/v1/sections")).hasStatusOk();
    }
}
