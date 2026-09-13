package com.tibell.trafficml;

import org.junit.jupiter.api.Test;

import com.tibell.trafficml.testsupport.AbstractIntegrationTest;

/**
 * Verifies the full Spring context (schedulers, JPA/Flyway, REST controllers, and all
 * three ingestion pipelines wired together) starts against a real PostGIS database.
 */
class BackendApplicationIT extends AbstractIntegrationTest {

    @Test
    void contextLoads() {
        // Intentionally empty: failure to start the context fails this test.
    }
}
