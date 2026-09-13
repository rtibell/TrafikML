package com.tibell.trafficml.repository;

import com.tibell.trafficml.entities.DefinitionOfSection;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DefinitionOfSectionRepository extends JpaRepository<DefinitionOfSection, Long> {
}
