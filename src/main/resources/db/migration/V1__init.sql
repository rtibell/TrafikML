-- TrafikML backend: initial schema for Stockholm (VST) road traffic data.
-- Requires a PostGIS-enabled PostgreSQL database.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- DefinitionOfSection: geographic definition of a road section (id = primary key).
-- ---------------------------------------------------------------------------
CREATE TABLE definition_of_section (
    id                            BIGINT PRIMARY KEY,
    name                          VARCHAR(255),
    region                        VARCHAR(50),
    county_no                     INTEGER,
    modified_time                 TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    geometry_modified_time        TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    activated                     BOOLEAN,
    activated_modified_time       TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    average_functional_road_class INTEGER,
    imported_time                 TIMESTAMP WITHOUT TIME ZONE,
    geometry                      GEOMETRY(LineString, 4326),
    record_created                TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    record_updated                TIMESTAMP WITHOUT TIME ZONE
);

CREATE INDEX idx_definition_of_section_geometry ON definition_of_section USING GIST (geometry);

-- ---------------------------------------------------------------------------
-- SpeedOfSection: periodic speed/status measurement for a section.
-- Composite key (section_id, measure_time); section_id is also a FK to
-- definition_of_section(id).
-- ---------------------------------------------------------------------------
CREATE TABLE speed_of_section (
    section_id      BIGINT NOT NULL REFERENCES definition_of_section (id),
    measure_time    TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    status          VARCHAR(50),
    speed           INTEGER,
    record_created  TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    record_updated  TIMESTAMP WITHOUT TIME ZONE,
    PRIMARY KEY (section_id, measure_time)
);

CREATE INDEX idx_speed_of_section_measure_time ON speed_of_section (measure_time);

-- ---------------------------------------------------------------------------
-- TrafficMessage: append-only feed of traffic incidents/roadworks (id = primary key).
-- ---------------------------------------------------------------------------
CREATE TABLE traffic_message (
    id                     BIGINT PRIMARY KEY,
    region                 VARCHAR(50),
    title                  VARCHAR(500),
    message                TEXT,
    provider               VARCHAR(255),
    message_type           VARCHAR(100),
    message_code           VARCHAR(100),
    traffic_type           VARCHAR(100),
    wgs84_position         GEOMETRY(Point, 4326),
    swe_ref_99_extent      GEOMETRY(LineString, 3006),
    affected_direction     VARCHAR(50),
    start_time             TIMESTAMP WITHOUT TIME ZONE,
    end_time               TIMESTAMP WITHOUT TIME ZONE,
    scheduled_occurrences  JSONB,
    version_time           TIMESTAMP WITHOUT TIME ZONE,
    icon_id                INTEGER,
    severity               INTEGER,
    is_future              BOOLEAN,
    road_closed            BOOLEAN,
    details_path           VARCHAR(1000),
    sort_index             INTEGER,
    severity_text          VARCHAR(255),
    work_state             INTEGER,
    record_created         TIMESTAMP WITHOUT TIME ZONE NOT NULL
);

CREATE INDEX idx_traffic_message_wgs84_position ON traffic_message USING GIST (wgs84_position);
CREATE INDEX idx_traffic_message_start_time ON traffic_message (start_time);
