package com.tibell.trafficml.testsupport;

import org.junit.jupiter.api.Tag;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

/**
 * Base class for tests that need a real, PostGIS-enabled PostgreSQL database (repository
 * and full-context tests).
 *
 * <p>The container is started once, in a static initializer, and never stopped
 * (Testcontainers' Ryuk side-car removes it when the JVM exits) - the "singleton
 * container" pattern. Using {@code @Testcontainers}/{@code @Container} instead would
 * start *and stop* this same static field around every subclass, since JUnit manages
 * container lifecycle per test class regardless of the field being shared; restarting an
 * already-stopped container that way is what caused intermittent
 * "connection refused" failures across the different IT classes.
 */
@Tag("integration")
@SpringBootTest
public abstract class AbstractIntegrationTest {

    // imresamu/postgis is a multi-arch (amd64 + arm64) rebuild of postgis/postgis, which
    // only ships amd64 images; this keeps tests runnable on Apple Silicon too.
    @ServiceConnection
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>(
            DockerImageName.parse("imresamu/postgis:17-3.5").asCompatibleSubstituteFor("postgres"));

    static {
        POSTGRES.start();
    }
}
