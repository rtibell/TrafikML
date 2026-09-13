package com.tibell.trafficml.entities;

import com.tibell.trafficml.services.TrafficMessageIngestionService;

import java.time.Instant;
import java.time.LocalDateTime;

import org.hibernate.annotations.ColumnTransformer;
import org.locationtech.jts.geom.LineString;
import org.locationtech.jts.geom.Point;

import tools.jackson.databind.JsonNode;

import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import com.tibell.trafficml.entities.JsonNodeConverter;

/**
 * A single traffic message (incident, roadwork, ...) as published by trafiken.nu.
 * Messages are append-only: the service only ever inserts, never updates
 * (see {@link TrafficMessageIngestionService}), so unlike the other two entities there
 * is no {@code recordUpdated}.
 */
@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "traffic_message")
public class TrafficMessage {

    /** Upstream message id; already numeric upstream, still normalized to {@code Long}. */
    @Id
    @Column(name = "id")
    private Long id;

    @Column(name = "region", length = 50)
    private String region;

    @Column(name = "title", length = 500)
    private String title;

    @Column(name = "message", columnDefinition = "text")
    private String message;

    @Column(name = "provider")
    private String provider;

    @Column(name = "message_type", length = 100)
    private String messageType;

    @Column(name = "message_code", length = 100)
    private String messageCode;

    @Column(name = "traffic_type", length = 100)
    private String trafficType;

    @Column(name = "wgs84_position", columnDefinition = "geometry(Point,4326)")
    private Point wgs84Position;

    @Column(name = "swe_ref_99_extent", columnDefinition = "geometry(LineString,3006)")
    private LineString sweRef99Extent;

    @Column(name = "affected_direction", length = 50)
    private String affectedDirection;

    @Column(name = "start_time")
    private LocalDateTime startTime;

    @Column(name = "end_time")
    private LocalDateTime endTime;

    @Convert(converter = JsonNodeConverter.class)
    @Column(name = "scheduled_occurrences", columnDefinition = "jsonb")
    // Postgres won't implicitly cast a varchar bind parameter to jsonb; without this the
    // JDBC driver sends the converted String as varchar and every insert/update fails.
    @ColumnTransformer(write = "?::jsonb")
    private JsonNode scheduledOccurrences;

    @Column(name = "version_time")
    private LocalDateTime versionTime;

    @Column(name = "icon_id")
    private Integer iconId;

    @Column(name = "severity")
    private Integer severity;

    @Column(name = "is_future")
    private Boolean isFuture;

    @Column(name = "road_closed")
    private Boolean roadClosed;

    @Column(name = "details_path", length = 1000)
    private String detailsPath;

    @Column(name = "sort_index")
    private Integer sortIndex;

    @Column(name = "severity_text")
    private String severityText;

    @Column(name = "work_state")
    private Integer workState;

    @Column(name = "record_created", nullable = false, updatable = false)
    private Instant recordCreated;

    public TrafficMessage(Long id) {
        this.id = id;
    }
}
