package com.tibell.trafficml.repository;

import com.tibell.trafficml.entities.TrafficMessage;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TrafficMessageRepository extends JpaRepository<TrafficMessage, Long> {
}
