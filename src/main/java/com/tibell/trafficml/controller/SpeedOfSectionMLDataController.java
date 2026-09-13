package com.tibell.trafficml.controller;

import com.tibell.trafficml.model.speedofsection.MachineLearningSpeedOfSectionView;
import com.tibell.trafficml.services.SpeedOfSectionMLData;

import java.util.List;

import org.springframework.data.domain.Pageable;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/sections/{id}/ml-speed-data")
public class SpeedOfSectionMLDataController {

    private final SpeedOfSectionMLData mlData;

    public SpeedOfSectionMLDataController(SpeedOfSectionMLData mlData) {
        this.mlData = mlData;
    }

    @GetMapping
    public List<MachineLearningSpeedOfSectionView> mlSpeedData(@PathVariable Long id, Pageable pageable) {
        return mlData.forSection(id, pageable);
    }
}
