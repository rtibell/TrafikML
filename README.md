# TrafficML Backend

## 1. Description of solution

TrafficML is a Spring Boot backend that continuously gathers road-traffic data for the
Stockholm region (VST) from the public [trafiken.nu](https://trafiken.nu) API and stores
it in a PostGIS-enabled PostgreSQL database. The collected data — section speeds, section
geometries, and traffic incident messages — is the raw material for a downstream Machine
Learning service that will predict when and where traffic congestion is likely to occur.

The backend runs three independent, scheduled ingestion pipelines, each polling one
upstream endpoint on its own cadence:

| Pipeline | Upstream endpoint | Default schedule | Persistence rule |
|---|---|---|---|
| **SpeedOfSection** | `GET /api/traveltime?region=vst` | every minute, 07:00–19:00 | Upsert keyed on the composite `(id, measureTime)`. `id` is also a real foreign key to `DefinitionOfSection`. |
| **DefinitionOfSection** | `GET /api/traveltime/action/getsections?region=vst` | once daily at 06:00 | Upsert keyed on `id`; a row is only overwritten when `modifiedTime` or `geometryModifiedTime` has moved on. |
| **TrafficMessage** | `GET /api/trafficmessages?region=vst&trafficType=Vägtrafik` | every 10 minutes | Insert-only, keyed on `id`. Every newly-inserted message is also rendered from a configurable template, written to `<storage-dir>/<id>.dat`, and posted to Slack via an incoming webhook. |

Each pipeline follows the same shape: an HTTP client fetches the raw JSON, a mapper
converts it to JPA entities (converting geometry to JTS/PostGIS types along the way), an
ingestion service decides insert vs. update vs. no-op and stamps `recordCreated` /
`recordUpdated`, and a `@Scheduled` method drives the whole thing on the configured cron
expression. A small set of read-only REST endpoints exposes the collected data as JSON,
independent of the ingestion jobs, satisfying the requirement that the service also be a
REST API.

All dynamic and environment-specific configuration — upstream URLs, cron expressions,
the Slack webhook/channel, the message template, the file storage directory, database
credentials — lives in `application.yml` / `application-{dev,test,prod}.yml`. Nothing of
that nature is hard-coded in Java source.

### Key design decisions

- **`SpeedOfSection.id` is a real JPA foreign key** (`@ManyToOne` + `@MapsId`) into
  `DefinitionOfSection`, so the database enforces the relationship described in the spec.
  Since definitions refresh once a day while speeds refresh every minute, a speed reading
  for a not-yet-known section id is logged and skipped rather than failing the whole
  batch.
- **`TrafficMessage.scheduledOccurrences`** has an undocumented, variable shape upstream,
  so it is stored as a `jsonb` column via a small `AttributeConverter` rather than being
  modelled field-by-field.
- **Traffic-message persistence is split from its side effects**: `TrafficMessagePersister`
  (the `@Transactional` database write) is a separate bean from
  `TrafficMessageIngestionService` (which renders the template, writes the file, and
  calls Slack). This means the database transaction commits before a slow or failing
  webhook call is made, and a Slack outage can never roll back a successful insert.
- **Geometry** is mapped with Hibernate Spatial / JTS: `Point` and `LineString` columns in
  PostGIS, matching the GeoJSON shapes returned by trafiken.nu (WGS84 for
  `geometry`/`wgs84Position`, SWEREF 99 TM for `sweRef99Extent`).

## 2. Description of project structure

The codebase is organized **by technical layer**, not by feature, per the target package
layout:

```
com.tibell.trafficml
├── BackendApplication.java        Spring Boot entry point (@SpringBootApplication)
│
├── configuration/                 @ConfigurationProperties bindings and Spring @Bean wiring
│   ├── TrafficMlProperties.java       binds the trafficml.* tree in application.yml
│   └── RestClientConfig.java          shared RestClient.Builder / timeouts
│
├── controller/                    @RestController / @RestControllerAdvice — the HTTP API
│   ├── DefinitionOfSectionController.java
│   ├── SpeedOfSectionController.java
│   ├── TrafficMessageController.java
│   └── GlobalExceptionHandler.java
│
├── model/                         Plain DTOs / POJOs — request & response shapes, not persisted
│   ├── PointGeoJson.java              shared GeoJSON DTOs (used by 2+ feeds)
│   ├── LineStringGeoJson.java
│   ├── ApiErrorResponse.java           REST error body
│   ├── definitionofsection/
│   │   ├── DefinitionOfSectionRawDto.java     raw upstream JSON shape
│   │   ├── DefinitionOfSectionPropertiesDto.java
│   │   └── DefinitionOfSectionView.java       read-only API response shape
│   ├── speedofsection/
│   │   ├── SpeedOfSectionRawDto.java
│   │   └── SpeedOfSectionView.java
│   └── trafficmessage/
│       ├── TrafficMessageRawDto.java
│       └── TrafficMessageView.java
│
├── entities/                      @Entity / @Embeddable / @MappedSuperclass — persisted state
│   ├── AuditableEntity.java            shared recordCreated/recordUpdated columns
│   ├── DefinitionOfSection.java
│   ├── SpeedOfSection.java
│   ├── SpeedOfSectionId.java           @Embeddable composite key (sectionId, measureTime)
│   ├── TrafficMessage.java
│   └── JsonNodeConverter.java          jsonb <-> JsonNode AttributeConverter
│
├── repository/                    Spring Data JPA repositories
│   ├── DefinitionOfSectionRepository.java
│   ├── SpeedOfSectionRepository.java
│   └── TrafficMessageRepository.java
│
├── mapper/                        Raw DTO <-> entity conversion, incl. GeoJSON -> JTS geometry
│   ├── DefinitionOfSectionMapper.java
│   ├── SpeedOfSectionMapper.java
│   ├── TrafficMessageMapper.java
│   └── GeoJsonConverter.java           GeoJSON DTO -> JTS Point/LineString
│
├── services/                      Business logic: HTTP clients, ingestion, side effects
│   ├── DefinitionOfSectionClient.java       fetches from trafiken.nu
│   ├── SpeedOfSectionClient.java
│   ├── TrafficMessageClient.java
│   ├── DefinitionOfSectionIngestionService.java   insert/update decision + persistence
│   ├── SpeedOfSectionIngestionService.java
│   ├── TrafficMessageIngestionService.java        orchestrates persist + template + file + Slack
│   ├── TrafficMessagePersister.java               the @Transactional insert-only write
│   ├── MessageTemplateRenderer.java               {{placeholder}} template rendering
│   ├── TrafficMessageFileStorageService.java      writes <id>.dat files
│   └── SlackNotificationClient.java               posts to the Slack webhook
│
└── scheduler/                     @Scheduled entry points, one per pipeline
    ├── DefinitionOfSectionScheduler.java
    ├── SpeedOfSectionScheduler.java
    └── TrafficMessageScheduler.java
```

A few classes don't map to an obviously-named package in the requested taxonomy; they
were placed as follows, on the reasoning noted:

| Class | Package | Reasoning |
|---|---|---|
| `DefinitionOfSectionClient`, `SpeedOfSectionClient`, `TrafficMessageClient`, `SlackNotificationClient` | `services` | They are business logic (fetching/pushing external data), not controllers, repositories, or mappers. |
| `GeoJsonConverter` | `mapper` | Its only job is converting one representation (GeoJSON DTO) into another (JTS geometry) — the definition of a mapper. |
| `ApiErrorResponse` | `model` | It's a DTO — the JSON shape of an error response — not a persisted entity. |
| `GlobalExceptionHandler` | `controller` | A `@RestControllerAdvice` is part of the HTTP/web layer alongside the controllers it supports. |
| `AuditableEntity`, `JsonNodeConverter` | `entities` | Both are JPA persistence support classes (a `@MappedSuperclass` and an `AttributeConverter`) used only by the entities. |
| `*View` records (e.g. `DefinitionOfSectionView`) | `model.<feature>` | They are response DTOs, kept alongside the other DTOs for that feed rather than in `controller`. |

Test sources mirror this same package layout under `src/test/java/com/tibell/trafficml`,
plus a `testsupport` package for shared test infrastructure (`AbstractIntegrationTest`,
`TestProperties`).

Non-Java layout:

```
build.gradle, settings.gradle, gradle.properties    Gradle build (Groovy DSL)
gradle/wrapper/                                     Gradle 9 wrapper (no local Gradle install needed)
docker-compose.yml                                  local PostGIS database for development
src/main/resources/
├── application.yml                                 shared defaults (URLs, cron, template, ...)
├── application-dev.yml                              local development overrides
├── application-test.yml                             shared test/staging environment overrides
├── application-prod.yml                             production overrides (no secrets defaulted)
└── db/migration/V1__init.sql                        Flyway schema (tables, indexes, PostGIS extension)
src/test/resources/application.yml                  test-run defaults (scheduling & Slack disabled)
```

## 3. How to build the application

The project uses the **Gradle wrapper**, so no local Gradle install is required — the
wrapper script downloads the exact Gradle version (9.5.1) the first time it runs.

```bash
# from the project root
./gradlew build
```

This compiles both source sets, runs the full test suite (unit tests plus Testcontainers
integration tests — Docker must be running, see §4), and produces an executable jar at
`build/libs/trafficml-backend-0.1.0.jar`.

Useful sub-tasks:

```bash
./gradlew compileJava        # compile production code only
./gradlew test               # run the test suite only
./gradlew bootJar            # build the jar without running tests
./gradlew clean build        # rebuild from scratch
```

The Java toolchain is pinned to **Java 25** in `build.gradle`; if a matching JDK isn't
already installed, Gradle's Foojay toolchain resolver (configured in `settings.gradle`)
downloads one automatically the first time you build — this requires outbound internet
access on that first run.

## 4. How to install the application and required tools

### Required tooling

| Tool | Version | Why | Install |
|---|---|---|---|
| **JDK** | 25 or later | Language/runtime baseline | Not strictly required to install manually — the Gradle toolchain resolver downloads JDK 25 automatically on first build. To install one yourself: `sdk install java 25-tem` (SDKMAN) or `brew install openjdk@25`. |
| **Git** *(optional)* | any | Cloning the repository | `brew install git` / OS package manager |
| **Docker Desktop** (or another Docker-compatible engine) | recent | Runs the local PostGIS database (`docker-compose.yml`) and the Testcontainers-based integration tests | https://www.docker.com/products/docker-desktop |

Gradle itself does **not** need to be installed — the checked-in wrapper
(`./gradlew` / `gradlew.bat`) takes care of it.

> **Apple Silicon (arm64) note:** the official `postgis/postgis` Docker image only
> publishes `amd64` builds. `docker-compose.yml` and the integration tests instead use
> `imresamu/postgis`, a multi-arch rebuild of the same image, so everything also runs
> natively on Apple Silicon without emulation.

### Installation steps

1. Install Docker Desktop and JDK 25 (or skip the JDK install and let the Gradle
   toolchain resolver fetch it on first build).
2. Clone or copy this repository and `cd` into it.
3. Verify the wrapper works and toolchain/dependencies resolve:
   ```bash
   ./gradlew --version
   ```
4. Build the project (see §3) to fetch all dependencies and compile:
   ```bash
   ./gradlew build
   ```

No separate installation step is needed for PostgreSQL/PostGIS — it runs as a Docker
container (§5) rather than a system service.

## 5. How to run the application

### 5.1 Start a database

```bash
docker compose up -d
```

This starts a PostGIS-enabled PostgreSQL 17 instance on `localhost:5432` with database
`trafficml_dev`, user `trafficml`, password `trafficml` (matching the defaults baked into
`application-dev.yml`). Flyway creates the schema automatically the first time the
application connects.

### 5.2 Run the application

```bash
./gradlew bootRun --args='--spring.profiles.active=dev'
```

or, using the built jar:

```bash
./gradlew bootJar
java -jar build/libs/trafficml-backend-0.1.0.jar --spring.profiles.active=dev
```

With the `dev` profile: Slack notifications are disabled by default (so local runs don't
spam a real channel), logging is verbose, and rendered traffic-message files are written
under `./data/messages/dev`.

Once running:
- The three schedulers start immediately (`SpeedOfSection` only fires between 07:00 and
  19:00 local time; the other two run continuously).
- The REST API is available at `http://localhost:8080/api/v1/...`
  (`/sections`, `/sections/{id}`, `/sections/{id}/speed`, `/traffic-messages`,
  `/traffic-messages/{id}`).
- Health/info endpoints are available at `http://localhost:8080/actuator/health` and
  `/actuator/info`.

### 5.3 Choosing an environment

Activate a profile with `--spring.profiles.active=<dev|test|prod>` (or the
`SPRING_PROFILES_ACTIVE` environment variable). `test` and `prod` expect their datasource
and Slack settings to come from environment variables rather than the built-in `dev`
defaults:

| Variable | Used by | Purpose |
|---|---|---|
| `TRAFFICML_DB_URL`, `TRAFFICML_DB_USERNAME`, `TRAFFICML_DB_PASSWORD` | test, prod | Database connection (prod has no defaults — it will fail fast if unset) |
| `TRAFFICML_SLACK_ENABLED` | all | Master on/off switch for Slack notifications |
| `TRAFFICML_SLACK_WEBHOOK_URL` | all | Slack incoming webhook URL |
| `TRAFFICML_SLACK_CHANNEL` | all | Slack channel name to post to |
| `TRAFFICML_MESSAGE_DIR` | all | Directory rendered `<id>.dat` traffic-message files are written to |

### 5.4 Running the tests

```bash
./gradlew test
```

Unit tests (mappers, template rendering, schedulers with mocked collaborators, HTTP
clients against `MockRestServiceServer`) need no external services. Tests tagged
`integration` additionally spin up a real PostGIS container via Testcontainers to exercise
JPA repositories and the full Spring context — Docker must be running for these.

## 6. Requirements of the application

### Functional requirements

- Collect three categories of Stockholm-region (VST) traffic data from trafiken.nu on
  independent schedules: section speeds (every minute, 07:00–19:00), section definitions
  (once daily at 06:00), and traffic messages (every 10 minutes).
- For each fetch, decide whether a record is new or an update to an existing one, and
  persist accordingly:
  - **SpeedOfSection**: composite key `(id, measureTime)`; `id` doubles as a foreign key
    to `DefinitionOfSection`.
  - **DefinitionOfSection**: primary key `id`; updated when `modifiedTime` or
    `geometryModifiedTime` changes.
  - **TrafficMessage**: primary key `id`; insert-only.
  - On insert, stamp `recordCreated`; on update, stamp `recordUpdated`.
- For every newly-inserted traffic message: render a human-readable message from a
  configurable template, save it to disk as `<id>.dat`, and post it to a configured Slack
  channel via an incoming webhook.
- Expose the collected data over a JSON REST API.
- Keep all URLs, schedules, templates, storage locations, and Slack settings in
  environment-specific YAML configuration rather than hard-coded in source.
- Provide a full automated test suite (unit and integration) covering each service,
  mapper, and utility.

### Non-functional requirements

- **Language/runtime**: Java 25 or later.
- **Framework**: Spring Boot 4 or later, using Spring Web, exposing REST endpoints that
  return JSON.
- **Build tool**: Gradle (Groovy DSL), version 9 or later.
- **Database**: PostgreSQL with the PostGIS extension, for storing and querying
  geographic data (road-section geometries, incident locations).
- **Environments**: distinct development, test, and production configurations.
- **Code quality**: layered architecture (controller / service / repository / mapper /
  entity / model / scheduler / configuration), with a complete set of automated tests per
  service, utility, and function.
