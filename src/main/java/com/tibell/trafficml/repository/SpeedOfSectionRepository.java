package com.tibell.trafficml.repository;

import com.tibell.trafficml.entities.SpeedOfSection;
import com.tibell.trafficml.entities.SpeedOfSectionId;

import java.util.List;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SpeedOfSectionRepository extends JpaRepository<SpeedOfSection, SpeedOfSectionId> {

    List<SpeedOfSection> findByIdSectionIdOrderByIdMeasureTimeDesc(Long sectionId, Pageable pageable);
}
