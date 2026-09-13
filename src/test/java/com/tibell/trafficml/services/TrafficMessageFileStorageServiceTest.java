package com.tibell.trafficml.services;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.tibell.trafficml.testsupport.TestProperties;

class TrafficMessageFileStorageServiceTest {

    @TempDir
    Path tempDir;

    @Test
    void storesMessageAsIdDotDatFile() throws IOException {
        Path storageDir = tempDir.resolve("messages");
        TrafficMessageFileStorageService service = new TrafficMessageFileStorageService(
                withStorageDir(storageDir.toString()));

        Path file = service.store(11851210L, "hello world");

        assertThat(file.getFileName().toString()).isEqualTo("11851210.dat");
        assertThat(Files.readString(file)).isEqualTo("hello world");
    }

    @Test
    void createsStorageDirectoryIfMissing() {
        Path storageDir = tempDir.resolve("nested/does/not/exist/yet");
        TrafficMessageFileStorageService service = new TrafficMessageFileStorageService(
                withStorageDir(storageDir.toString()));

        service.store(1L, "content");

        assertThat(Files.isDirectory(storageDir)).isTrue();
    }

    private com.tibell.trafficml.configuration.TrafficMlProperties withStorageDir(String dir) {
        var base = TestProperties.withApiUrl("https://example.invalid");
        return new com.tibell.trafficml.configuration.TrafficMlProperties(
                base.api(), base.schedule(), base.slack(),
                new com.tibell.trafficml.configuration.TrafficMlProperties.TrafficMessage(base.trafficMessage().template(), dir));
    }
}
