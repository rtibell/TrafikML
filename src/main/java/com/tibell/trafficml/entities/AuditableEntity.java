package com.tibell.trafficml.entities;

import java.time.Instant;

import jakarta.persistence.Column;
import jakarta.persistence.MappedSuperclass;

import lombok.Getter;
import lombok.Setter;

/**
 * Shared "insert or update" bookkeeping for entities that are periodically re-fetched
 * from an upstream source: {@code recordCreated} is stamped once on first insert,
 * {@code recordUpdated} is refreshed every time an existing row is overwritten with
 * newer upstream data.
 */
@Getter
@Setter
@MappedSuperclass
public abstract class AuditableEntity {

    @Column(name = "record_created", nullable = false, updatable = false)
    private Instant recordCreated;

    @Column(name = "record_updated")
    private Instant recordUpdated;
}
