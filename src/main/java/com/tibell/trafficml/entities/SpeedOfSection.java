package com.tibell.trafficml.entities;

import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.MapsId;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import com.tibell.trafficml.entities.AuditableEntity;
import com.tibell.trafficml.entities.DefinitionOfSection;

/**
 * A single speed/status measurement for a road section at a point in time.
 * {@code id} (the section id) doubles as a foreign key into
 * {@link DefinitionOfSection}, and together with {@code measureTime} forms the
 * composite primary key used to decide insert vs. update.
 */
@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "speed_of_section")
public class SpeedOfSection extends AuditableEntity {

    @EmbeddedId
    private SpeedOfSectionId id;

    @ManyToOne(fetch = FetchType.LAZY)
    @MapsId("sectionId")
    @JoinColumn(name = "section_id", nullable = false)
    private DefinitionOfSection section;

    @Column(name = "status", length = 50)
    private String status;

    @Column(name = "speed")
    private Integer speed;

    public SpeedOfSection(DefinitionOfSection section, java.time.LocalDateTime measureTime) {
        this.section = section;
        this.id = new SpeedOfSectionId(section.getId(), measureTime);
    }
}
