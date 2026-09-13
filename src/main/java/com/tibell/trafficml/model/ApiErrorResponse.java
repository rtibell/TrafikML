package com.tibell.trafficml.model;

import java.time.Instant;

/**
 * Uniform JSON error body returned by the REST API on failure.
 */
public record ApiErrorResponse(Instant timestamp, int status, String error, String message, String path) {

    public static ApiErrorResponse of(int status, String error, String message, String path) {
        return new ApiErrorResponse(Instant.now(), status, error, message, path);
    }
}
