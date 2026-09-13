package com.tibell.trafficml.configuration;

import java.time.Duration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.ClientHttpRequestFactory;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

/**
 * Provides the {@link RestClient} used by every outbound HTTP integration (trafiken.nu
 * endpoints and the Slack webhook). A single, shared, sensibly-timed client keeps the
 * three scheduled services from hanging the scheduler thread pool on a slow upstream.
 */
@Configuration
public class RestClientConfig {

    @Bean
    public ClientHttpRequestFactory trafficMlClientHttpRequestFactory() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(5));
        factory.setReadTimeout(Duration.ofSeconds(15));
        return factory;
    }

    @Bean
    public RestClient.Builder trafficMlRestClientBuilder(ClientHttpRequestFactory trafficMlClientHttpRequestFactory) {
        return RestClient.builder().requestFactory(trafficMlClientHttpRequestFactory);
    }
}
