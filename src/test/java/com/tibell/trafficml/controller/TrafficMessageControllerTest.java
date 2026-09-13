package com.tibell.trafficml.controller;

import com.tibell.trafficml.entities.TrafficMessage;
import com.tibell.trafficml.repository.TrafficMessageRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.assertj.MockMvcTester;

@WebMvcTest(TrafficMessageController.class)
class TrafficMessageControllerTest {

    @Autowired
    private MockMvcTester mvc;

    @MockitoBean
    private TrafficMessageRepository repository;

    @Test
    void getReturnsMessageAsJson() {
        TrafficMessage message = new TrafficMessage(11851210L);
        message.setTitle("Test message");
        when(repository.findById(11851210L)).thenReturn(Optional.of(message));

        assertThat(mvc.get().uri("/api/v1/traffic-messages/11851210"))
                .hasStatusOk()
                .bodyJson()
                .extractingPath("$.title").isEqualTo("Test message");
    }

    @Test
    void getReturns404WhenMessageUnknown() {
        when(repository.findById(1L)).thenReturn(Optional.empty());

        assertThat(mvc.get().uri("/api/v1/traffic-messages/1")).hasStatus4xxClientError();
    }
}
