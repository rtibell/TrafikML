package com.tibell.trafficml.entities;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.NullNode;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;

/**
 * Persists arbitrarily-shaped JSON (e.g. {@code scheduledOccurrences}, whose element
 * structure is not documented by the upstream API) as a {@code jsonb} column without
 * losing information, instead of mapping it field-by-field.
 */
@Converter
public class JsonNodeConverter implements AttributeConverter<JsonNode, String> {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Override
    public String convertToDatabaseColumn(JsonNode attribute) {
        if (attribute == null || attribute.isNull()) {
            return null;
        }
        return MAPPER.writeValueAsString(attribute);
    }

    @Override
    public JsonNode convertToEntityAttribute(String dbData) {
        if (dbData == null || dbData.isBlank()) {
            return NullNode.getInstance();
        }
        return MAPPER.readTree(dbData);
    }
}
