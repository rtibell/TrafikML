package com.tibell.trafficml.services;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.springframework.stereotype.Service;

import com.tibell.trafficml.configuration.TrafficMlProperties;

/**
 * Writes the rendered message text to {@code <storageDir>/<id>.dat}, one file per
 * traffic message, as specified for the TrafficMessages service.
 */
@Service
public class TrafficMessageFileStorageService {

    private final Path storageDir;

    public TrafficMessageFileStorageService(TrafficMlProperties properties) {
        this.storageDir = Path.of(properties.trafficMessage().storageDir());
    }

    public Path store(Long id, String content) {
        try {
            Files.createDirectories(storageDir);
            Path file = storageDir.resolve(id + ".dat");
            Files.writeString(file, content, StandardCharsets.UTF_8);
            return file;
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to write message file for traffic message " + id, e);
        }
    }
}
