# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

TrafficML is a Spring Boot 4 (Java 25) backend that continuously gathers road-traffic
data for the Stockholm region (VST) from the public [trafiken.nu](https://trafiken.nu)
API and stores it in a PostGIS-enabled PostgreSQL database, as raw material for a
downstream ML service predicting traffic congestion. It runs three independent scheduled
ingestion pipelines and exposes the collected data (plus derived ML feature data) over a
JSON REST API. See `README.md` §1–§2 for the full narrative and `REST-opperations.md` for
REST endpoint documentation with curl examples.

## Commands

```bash
./gradlew build                # compile, run the full test suite, build the jar
./gradlew compileJava           # compile production code only
./gradlew test                  # run the full test suite (see "Testing" below — needs Docker)
./gradlew bootJar               # build the jar without running tests
./gradlew clean build           # rebuild from scratch

# single test class
./gradlew test --tests "com.tibell.trafficml.services.SpeedOfSectionMLDataTest"
# single test method
./gradlew test --tests "com.tibell.trafficml.services.SpeedOfSectionMLDataTest.mapsAllFourStatusValues"
# a whole package
./gradlew test --tests "com.tibell.trafficml.util.*"

docker compose up -d             # start local PostGIS (localhost:5432, db/user/pass: trafficml)
./gradlew bootRun --args='--spring.profiles.active=dev'   # run the app locally
```

No local Gradle/JDK install is required — `./gradlew` is the wrapper (Gradle 9.5.1), and
the Foojay toolchain resolver in `settings.gradle` downloads JDK 25 automatically on
first build if needed.

### Testing

`./gradlew test` runs unit tests and Testcontainers-backed integration tests
(`@Tag("integration")`, classes named `*IT`) together — there is no separate Gradle task
that skips one or the other, so **Docker must be running** for a full `./gradlew test`.
For fast, Docker-free iteration, target a specific unit test class/package with
`--tests` as shown above.

- Unit tests: plain JUnit 5, often `@ExtendWith(MockitoExtension.class)` with `@Mock`
  collaborators constructed in a `@BeforeEach` (not as a field initializer — Mockito
  injects `@Mock` fields *after* other field initializers run, so `private final Foo x =
  new Foo(mock)` sees a null mock; see any `*ServiceTest`).
- Controller tests: `@WebMvcTest(SomeController.class)` + `MockMvcTester` (AssertJ-style
  fluent assertions) + `@MockitoBean` for the service/repository dependency.
- HTTP client tests: `MockRestServiceServer` bound to a fresh `RestClient.Builder` (see
  `*ClientTest`), not a running server.
- Repository/full-context integration tests extend
  `testsupport.AbstractIntegrationTest`, which starts one `PostgreSQLContainer`
  (`imresamu/postgis` — a multi-arch rebuild of `postgis/postgis`, needed for Apple
  Silicon) in a **static initializer**, never stopped explicitly. Do not switch this to
  `@Testcontainers`/`@Container` — JUnit would then stop/restart the shared static
  container per test class, which previously caused intermittent "connection refused"
  failures.
- `testsupport.TestProperties.withApiUrl(url)` builds a minimal valid
  `TrafficMlProperties` for unit tests that construct a client/service by hand instead of
  through the Spring context.

## Architecture

The codebase is organized **by technical layer**, not by feature:

```
controller/    @RestController + one @RestControllerAdvice (GlobalExceptionHandler) — HTTP API
model/         DTOs: <feature>/*RawDto (raw upstream JSON), <feature>/*View (API response shape)
entities/      @Entity / @Embeddable / @MappedSuperclass — persisted state
repository/    Spring Data JPA repositories
mapper/        Raw DTO <-> entity conversion (incl. GeoJSON -> JTS geometry)
services/      Business logic: HTTP clients, ingestion decisions, side effects, ML feature derivation
scheduler/     @Scheduled entry points, one per ingestion pipeline
configuration/ @ConfigurationProperties bindings and shared Spring @Bean wiring
util/          Stateless helpers with no Spring dependency (e.g. SwedishHolidays)
```

Test sources mirror this exact package layout under `src/test/java/...`, plus
`testsupport/` for shared test infrastructure. See README.md §2 for the full
file-by-file listing and the reasoning behind a few classes' package placement.

### The three ingestion pipelines

Each of `SpeedOfSection`, `DefinitionOfSection`, `TrafficMessage` follows the same shape:
`*Client` (fetches raw JSON via `RestClient`) → `*Mapper` (raw DTO → entity) →
`*IngestionService` (insert/update/no-op decision, stamps `recordCreated`/
`recordUpdated`) → `*Scheduler` (`@Scheduled`, cron from `application.yml`). They run on
independent schedules and persist under different rules — see the table in README.md §1.
`SpeedOfSection.id` is a real JPA foreign key (`@ManyToOne` + `@MapsId`) into
`DefinitionOfSection`; a speed reading for a not-yet-known section id is logged and
skipped, not treated as a fatal error, since definitions refresh once daily but speeds
refresh every minute.

`TrafficMessage` ingestion additionally splits persistence from side effects:
`TrafficMessagePersister` (the `@Transactional` DB write) is a separate bean from
`TrafficMessageIngestionService` (which renders the message template, writes the
`<id>.dat` file, and posts to Slack) — so the DB transaction commits before a slow/failing
Slack webhook call, and a Slack outage never rolls back a successful insert.

### ML feature endpoint

`SpeedOfSectionMLDataController` (`GET /api/v1/sections/{id}/ml-speed-data`) forwards to
`SpeedOfSectionMLData`, which re-shapes `SpeedOfSection` history into
`MachineLearningSpeedOfSectionView` — numeric features (status enum, day-of-week number,
minutes since 06:00 daybreak wrapping past midnight, holiday encodings) consumed by the
downstream ML model. Holiday-related fields are derived from `util.SwedishHolidays`,
whose `nextHoliday(LocalDate)` returns the next holiday **strictly after** the given date
(so if the given date is itself a holiday, callers needing to detect that must additionally
check `nextHoliday(date.minusDays(1))`, as `SpeedOfSectionMLData` does). Field mappings
are documented in full in `REST-opperations.md`.

### Database schema sync

`spring.jpa.hibernate.ddl-auto` is `validate` (see `application.yml`) — Hibernate never
generates schema; it fails fast at startup if entities and the Flyway-managed schema
(`src/main/resources/db/migration/`) disagree, including on column length/nullability
(`@Column(length = …)`, `nullable = …`) and not just column presence. When changing an
entity's column mapping, add a new `V<n>__*.sql` Flyway migration rather than editing
`V1__init.sql` in place — once applied anywhere, Flyway checksums it. Verify new/changed
mappings against what trafiken.nu actually sends, not just the migration file: it has
shipped fields that are legitimately null for a large fraction of records even though the
original schema marked the column `NOT NULL` (see the `V2__*.sql` migration).

### Working with the upstream trafiken.nu API

- `GET /api/traveltime/action/getsections?region=vst` (`DefinitionOfSectionClient`)
  serves valid JSON but mislabels it `Content-Type: text/plain` — the other two upstream
  endpoints correctly send `application/json`. This is handled locally in
  `DefinitionOfSectionClient` (a cloned `RestClient.Builder` with a
  `JacksonJsonHttpMessageConverter` configured to also accept `text/plain`), not in the
  shared `trafficMlRestClientBuilder` bean, so the other two clients are unaffected.
- Jackson is Jackson 3 (`tools.jackson.*` for core/databind), but annotations still come
  from the classic `com.fasterxml.jackson.annotation` package — both imports are correct
  and expected side by side in the same file.
- All upstream URLs, cron expressions, and other environment-specific values live in
  `application.yml` / `application-{dev,test,prod}.yml` — never hard-code them in Java.
