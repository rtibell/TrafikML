package com.tibell.trafficml.services;

import com.tibell.trafficml.model.trafficmessage.TrafficMessageRawDto;

import java.net.URI;
import java.util.List;

import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import com.tibell.trafficml.configuration.TrafficMlProperties;

/**
 * Fetches current traffic messages (incidents, roadworks, ...) from trafiken.nu.
 */
@Component
public class TrafficMessageClient {

    private final RestClient restClient;
    private final URI uri;

    public TrafficMessageClient(RestClient.Builder trafficMlRestClientBuilder, TrafficMlProperties properties) {
        this.restClient = trafficMlRestClientBuilder.build();
        // URI.create() (rather than RestClient's uri(String) template method) treats the
        // configured URL as already fully-encoded, so its "%C3%A4" isn't re-escaped to "%25C3%25A4".
        this.uri = URI.create(properties.api().trafficMessagesUrl());
    }

    public List<TrafficMessageRawDto> fetchMessages() {
        TrafficMessageRawDto[] response = restClient.get()
                .uri(uri)
                .retrieve()
                .body(TrafficMessageRawDto[].class);
        return response == null ? List.of() : List.of(response);
    }
}
