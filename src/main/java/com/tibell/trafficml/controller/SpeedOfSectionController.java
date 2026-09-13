package com.tibell.trafficml.controller;

import com.tibell.trafficml.model.speedofsection.SpeedOfSectionView;
import com.tibell.trafficml.repository.SpeedOfSectionRepository;

import java.util.List;

import org.springframework.data.domain.Pageable;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/sections/{sectionId}/speed")
public class SpeedOfSectionController {

    private final SpeedOfSectionRepository repository;

    public SpeedOfSectionController(SpeedOfSectionRepository repository) {
        this.repository = repository;
    }

    @GetMapping
    public List<SpeedOfSectionView> history(@PathVariable Long sectionId, Pageable pageable) {
        return repository.findByIdSectionIdOrderByIdMeasureTimeDesc(sectionId, pageable).stream()
                .map(SpeedOfSectionView::from)
                .toList();
    }
}
