package com.tibell.trafficml.controller;

import com.tibell.trafficml.entities.DefinitionOfSection;
import com.tibell.trafficml.model.definitionofsection.DefinitionOfSectionView;
import com.tibell.trafficml.repository.DefinitionOfSectionRepository;

import java.util.List;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.persistence.EntityNotFoundException;

@RestController
@RequestMapping("/api/v1/sections")
public class DefinitionOfSectionController {

    private final DefinitionOfSectionRepository repository;

    public DefinitionOfSectionController(DefinitionOfSectionRepository repository) {
        this.repository = repository;
    }

    @GetMapping
    public List<DefinitionOfSectionView> list(Pageable pageable) {
        Page<DefinitionOfSection> page = repository.findAll(pageable);
        return page.map(DefinitionOfSectionView::from).getContent();
    }

    @GetMapping("/{id}")
    public DefinitionOfSectionView get(@PathVariable Long id) {
        return repository.findById(id)
                .map(DefinitionOfSectionView::from)
                .orElseThrow(() -> new EntityNotFoundException("No section definition found for id " + id));
    }
}
