package com.tibell.trafficml.testsupport;

import com.tibell.trafficml.configuration.TrafficMlProperties;

/**
 * Builds a valid, minimal {@link TrafficMlProperties} for unit tests that need to
 * construct a client/service by hand instead of through the Spring context.
 */
public final class TestProperties {

    private TestProperties() {
    }

    public static TrafficMlProperties withApiUrl(String url) {
        return new TrafficMlProperties(
                new TrafficMlProperties.Api(url, url, url),
                new TrafficMlProperties.Schedule(false, "Europe/Stockholm", "0 * 7-19 * * *", "0 0 6 * * *",
                        "0 */10 * * * *"),
                new TrafficMlProperties.Slack("https://hooks.slack.example.invalid/services/T/B/x", "#test", true),
                new TrafficMlProperties.TrafficMessage("{{id}}: {{title}}", "build/test-output/messages"));
    }
}
