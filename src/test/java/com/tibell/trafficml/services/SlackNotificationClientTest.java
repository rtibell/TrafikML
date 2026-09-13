package com.tibell.trafficml.services;

import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import com.tibell.trafficml.configuration.TrafficMlProperties;
import com.tibell.trafficml.testsupport.TestProperties;

class SlackNotificationClientTest {

    private static final String WEBHOOK_URL = "https://hooks.slack.example.invalid/services/T/B/x";

    @Test
    void postsRenderedMessageAndChannelToWebhook() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo(WEBHOOK_URL))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().json("""
                        {"channel": "#test", "text": "hello"}
                        """))
                .andRespond(withSuccess("ok", MediaType.TEXT_PLAIN));

        TrafficMlProperties properties = withSlack(new TrafficMlProperties.Slack(WEBHOOK_URL, "#test", true));
        new SlackNotificationClient(builder, properties).send("hello");

        server.verify();
    }

    @Test
    void doesNotCallWebhookWhenDisabled() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        // No expectations registered: any HTTP call would fail verification below.

        TrafficMlProperties properties = withSlack(new TrafficMlProperties.Slack(WEBHOOK_URL, "#test", false));
        new SlackNotificationClient(builder, properties).send("hello");

        server.verify();
    }

    @Test
    void doesNotCallWebhookWhenUrlIsBlank() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();

        TrafficMlProperties properties = withSlack(new TrafficMlProperties.Slack("", "#test", true));
        new SlackNotificationClient(builder, properties).send("hello");

        server.verify();
    }

    private TrafficMlProperties withSlack(TrafficMlProperties.Slack slack) {
        TrafficMlProperties base = TestProperties.withApiUrl("https://example.invalid");
        return new TrafficMlProperties(base.api(), base.schedule(), slack, base.trafficMessage());
    }
}
