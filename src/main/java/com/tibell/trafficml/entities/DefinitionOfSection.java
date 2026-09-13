package com.tibell.trafficml.entities;

import java.time.LocalDateTime;

import org.locationtech.jts.geom.LineString;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import com.tibell.trafficml.entities.AuditableEntity;

/**
 * A road section as defined by trafiken.nu: a named, geographically-shaped stretch of
 * road that {@link com.tibell.trafficml.entities.SpeedOfSection} measurements refer
 * to via {@code id} (foreign key).
 */
@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "definition_of_section")
public class DefinitionOfSection extends AuditableEntity {

    /** Upstream section id, converted from the JSON string to a {@code Long}. */
    @Id
    @Column(name = "id")
    private Long id;

    @Column(name = "name")
    private String name;

    @Column(name = "region", length = 50)
    private String region;

    @Column(name = "county_no")
    private Integer countyNo;

    @Column(name = "modified_time", nullable = false)
    private LocalDateTime modifiedTime;

    @Column(name = "geometry_modified_time", nullable = false)
    private LocalDateTime geometryModifiedTime;

    @Column(name = "activated")
    private Boolean activated;

    @Column(name = "activated_modified_time")
    private LocalDateTime activatedModifiedTime;

    @Column(name = "average_functional_road_class")
    private Integer averageFunctionalRoadClass;

    @Column(name = "imported_time")
    private LocalDateTime importedTime;

    @Column(name = "geometry", columnDefinition = "geometry(LineString,4326)")
    private LineString geometry;

    public DefinitionOfSection(Long id) {
        this.id = id;
    }
}
