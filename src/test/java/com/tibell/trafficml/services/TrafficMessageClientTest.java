package com.tibell.trafficml.services;

import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import com.tibell.trafficml.testsupport.TestProperties;

class TrafficMessageClientTest {

    private static final String URL = "https://trafiken.nu/api/trafficmessages?region=vst&trafficType=V%C3%A4gtrafik";

    @Test
    void fetchMessagesParsesJsonArrayResponse() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo(URL))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess("""
                        [{
                          "id": 11851210,
                          "region": "VST",
                          "title": "Test message",
                          "message": "<span>Text</span>",
                          "wgs84Position": {"type": "Point", "coordinates": [18.09, 59.57]},
                          "sweRef99Extent": {"type": "LineString", "coordinates": [[675188.39, 6605920.66]]},
                          "startTime": "2026-09-14T07:00:00",
                          "endTime": "2026-10-12T16:00:00",
                          "scheduledOccurrences": [],
                          "versionTime": "2026-09-11T13:17:26",
                          "severity": 2,
                          "isFuture": true,
                          "roadClosed": false,
                          "sortIndex": 4347,
                          "severityText": "Liten paverkan",
                          "workState": 0
                        }]
                        """, MediaType.APPLICATION_JSON));

        TrafficMessageClient client = new TrafficMessageClient(builder, TestProperties.withApiUrl(URL));

        List<TrafficMessageRawDto> messages = client.fetchMessages();

        assertThat(messages).hasSize(1);
        assertThat(messages.get(0).id()).isEqualTo(11851210L);
        assertThat(messages.get(0).title()).isEqualTo("Test message");
        server.verify();
    }
}
