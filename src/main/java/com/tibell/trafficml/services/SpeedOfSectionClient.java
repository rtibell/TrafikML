package com.tibell.trafficml.services;

import com.tibell.trafficml.model.speedofsection.SpeedOfSectionRawDto;

import java.net.URI;
import java.util.List;

import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import com.tibell.trafficml.configuration.TrafficMlProperties;

/**
 * Fetches the current speed/status snapshot for all sections from trafiken.nu.
 */
@Component
public class SpeedOfSectionClient {

    private final RestClient restClient;
    private final URI uri;

    public SpeedOfSectionClient(RestClient.Builder trafficMlRestClientBuilder, TrafficMlProperties properties) {
        this.restClient = trafficMlRestClientBuilder.build();
        this.uri = URI.create(properties.api().speedOfSectionUrl());
    }

    public List<SpeedOfSectionRawDto> fetchSpeeds() {
        SpeedOfSectionRawDto[] response = restClient.get()
                .uri(uri)
                .retrieve()
                .body(SpeedOfSectionRawDto[].class);
        return response == null ? List.of() : List.of(response);
    }
}
