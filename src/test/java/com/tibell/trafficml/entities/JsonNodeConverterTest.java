package com.tibell.trafficml.entities;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

class JsonNodeConverterTest {

    private final JsonNodeConverter converter = new JsonNodeConverter();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void roundTripsAJsonArray() {
        JsonNode original = objectMapper.valueToTree(java.util.List.of(java.util.Map.of("from", "08:00")));

        String column = converter.convertToDatabaseColumn(original);
        JsonNode restored = converter.convertToEntityAttribute(column);

        assertThat(restored).isEqualTo(original);
    }

    @Test
    void nullAttributeConvertsToNullColumn() {
        assertThat(converter.convertToDatabaseColumn(null)).isNull();
    }

    @Test
    void blankColumnConvertsToNullNode() {
        JsonNode restored = converter.convertToEntityAttribute(null);

        assertThat(restored.isNull()).isTrue();
    }
}
