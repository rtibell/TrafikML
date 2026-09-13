package com.tibell.trafficml.services;

import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionRawDto;

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

class DefinitionOfSectionClientTest {

    private static final String URL = "https://trafiken.nu/api/traveltime/action/getsections?region=vst";

    @Test
    void fetchSectionsParsesGeoJsonFeatureCollectionResponse() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo(URL))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess("""
                        [{
                          "type": "Feature",
                          "id": "45553",
                          "geometry": {"type": "LineString", "coordinates": [[17.9, 58.9], [17.91, 58.91]]},
                          "properties": {
                            "name": "Test section",
                            "region": "VST",
                            "countyNo": 1,
                            "modifiedTime": "2026-05-21T08:30:52",
                            "geometryModifiedTime": "2026-03-11T03:29:52",
                            "activated": true,
                            "averageFunctionalRoadClass": 3,
                            "importedTime": "2026-05-21T10:31:12.6484827"
                          }
                        }]
                        """, MediaType.APPLICATION_JSON));

        DefinitionOfSectionClient client = new DefinitionOfSectionClient(builder, TestProperties.withApiUrl(URL));

        List<DefinitionOfSectionRawDto> sections = client.fetchSections();

        assertThat(sections).hasSize(1);
        assertThat(sections.get(0).id()).isEqualTo("45553");
        assertThat(sections.get(0).properties().name()).isEqualTo("Test section");
        server.verify();
    }

    @Test
    void fetchSectionsParsesResponseMislabeledAsTextPlain() {
        // trafiken.nu serves this endpoint's JSON body with a "text/plain" Content-Type
        // header (unlike its other endpoints, which send "application/json").
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo(URL))
                .andExpect(method(HttpMethod.GET))
                .andRespond(withSuccess("""
                        [{
                          "type": "Feature",
                          "id": "45553",
                          "geometry": {"type": "LineString", "coordinates": [[17.9, 58.9], [17.91, 58.91]]},
                          "properties": {
                            "name": "Test section",
                            "region": "VST",
                            "countyNo": 1,
                            "modifiedTime": "2026-05-21T08:30:52",
                            "geometryModifiedTime": "2026-03-11T03:29:52",
                            "activated": true,
                            "averageFunctionalRoadClass": 3,
                            "importedTime": "2026-05-21T10:31:12.6484827"
                          }
                        }]
                        """, MediaType.TEXT_PLAIN));

        DefinitionOfSectionClient client = new DefinitionOfSectionClient(builder, TestProperties.withApiUrl(URL));

        List<DefinitionOfSectionRawDto> sections = client.fetchSections();

        assertThat(sections).hasSize(1);
        assertThat(sections.get(0).id()).isEqualTo("45553");
        server.verify();
    }
}
