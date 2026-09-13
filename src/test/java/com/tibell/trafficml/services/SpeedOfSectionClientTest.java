package com.tibell.trafficml.services;

import com.tibell.trafficml.model.speedofsection.SpeedOfSectionRawDto;

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

class SpeedOfSectionClientTest {

    private static final String URL = "https://trafiken.nu/api/traveltime?region=vst";

    @Test
    void fetchSpeedsParsesJsonArrayResponse() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo(URL))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess("""
                        [{"id": "44469", "status": "freeflow", "speed": 48, "measureTime": "2026-09-12T17:19:00"}]
                        """, MediaType.APPLICATION_JSON));

        SpeedOfSectionClient client = new SpeedOfSectionClient(builder, TestProperties.withApiUrl(URL));

        List<SpeedOfSectionRawDto> speeds = client.fetchSpeeds();

        assertThat(speeds).hasSize(1);
        assertThat(speeds.get(0).id()).isEqualTo("44469");
        assertThat(speeds.get(0).speed()).isEqualTo(48);
        server.verify();
    }

    @Test
    void fetchSpeedsReturnsEmptyListWhenBodyIsEmptyArray() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo(URL)).andRespond(withSuccess("[]", MediaType.APPLICATION_JSON));

        List<SpeedOfSectionRawDto> speeds = new SpeedOfSectionClient(builder, TestProperties.withApiUrl(URL)).fetchSpeeds();

        assertThat(speeds).isEmpty();
    }
}
