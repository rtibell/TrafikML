package com.tibell.trafficml.controller;

import com.tibell.trafficml.entities.TrafficMessage;
import com.tibell.trafficml.model.trafficmessage.TrafficMessageView;
import com.tibell.trafficml.repository.TrafficMessageRepository;

import java.util.List;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.persistence.EntityNotFoundException;

@RestController
@RequestMapping("/api/v1/traffic-messages")
public class TrafficMessageController {

    private final TrafficMessageRepository repository;

    public TrafficMessageController(TrafficMessageRepository repository) {
        this.repository = repository;
    }

    @GetMapping
    public List<TrafficMessageView> list(Pageable pageable) {
        Page<TrafficMessage> page = repository.findAll(pageable);
        return page.map(TrafficMessageView::from).getContent();
    }

    @GetMapping("/{id}")
    public TrafficMessageView get(@PathVariable Long id) {
        return repository.findById(id)
                .map(TrafficMessageView::from)
                .orElseThrow(() -> new EntityNotFoundException("No traffic message found for id " + id));
    }
}
