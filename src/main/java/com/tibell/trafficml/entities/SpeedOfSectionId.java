package com.tibell.trafficml.entities;

import java.io.Serializable;
import java.time.LocalDateTime;

import jakarta.persistence.Embeddable;

/**
 * Composite key {@code (id, measureTime)} identifying a single speed measurement, as
 * specified for the SpeedOfSection service.
 */
@Embeddable
public record SpeedOfSectionId(Long sectionId, LocalDateTime measureTime) implements Serializable {
}
