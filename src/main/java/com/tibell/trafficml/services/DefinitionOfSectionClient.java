package com.tibell.trafficml.services;

import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionRawDto;

import java.net.URI;
import java.util.List;

import org.springframework.http.MediaType;
import org.springframework.http.converter.json.JacksonJsonHttpMessageConverter;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import com.tibell.trafficml.configuration.TrafficMlProperties;

/**
 * Fetches the current set of section definitions from trafiken.nu.
 */
@Component
public class DefinitionOfSectionClient {

    private final RestClient restClient;
    private final URI uri;

    public DefinitionOfSectionClient(RestClient.Builder trafficMlRestClientBuilder, TrafficMlProperties properties) {
        // This endpoint serves a JSON body but mislabels it as "text/plain"; the other
        // trafiken.nu endpoints send proper "application/json", so this is fixed here
        // (on a cloned builder) rather than on the shared trafficMlRestClientBuilder.
        JacksonJsonHttpMessageConverter jsonConverter = new JacksonJsonHttpMessageConverter();
        jsonConverter.setSupportedMediaTypes(List.of(MediaType.APPLICATION_JSON, MediaType.TEXT_PLAIN));
        this.restClient = trafficMlRestClientBuilder.clone()
                .configureMessageConverters(converters -> converters.withJsonConverter(jsonConverter))
                .build();
        this.uri = URI.create(properties.api().definitionOfSectionUrl());
    }

    public List<DefinitionOfSectionRawDto> fetchSections() {
        DefinitionOfSectionRawDto[] response = restClient.get()
                .uri(uri)
                .retrieve()
                .body(DefinitionOfSectionRawDto[].class);
        return response == null ? List.of() : List.of(response);
    }
}
