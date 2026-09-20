# REST API operations

Base URL for a local run (default `server.port` from `application.yml`):

```
http://localhost:8080
```

All responses are JSON. All endpoints are read-only (`GET`) and query data already
collected by the three ingestion pipelines described in `README.md` — they do not trigger
an upstream fetch.

On error, every endpoint returns a uniform error body:

```json
{
  "timestamp": "2026-09-13T13:44:00.123Z",
  "status": 404,
  "error": "Not Found",
  "message": "No section definition found for id 999999",
  "path": "/api/v1/sections/999999"
}
```

`404 Not Found` is returned for an unknown single-resource id; `400 Bad Request` for a
malformed argument (e.g. a non-numeric id or an invalid `Pageable` parameter).

---

## Section definitions — `/api/v1/sections`

### `GET /api/v1/sections`

Paginated list of all known road section definitions (geometry, name, region, ...).

**Query parameters** (standard Spring Data `Pageable` binding):

| Parameter | Description | Default |
|---|---|---|
| `page` | Zero-based page index | `0` |
| `size` | Page size | `20` |
| `sort` | `property,(asc\|desc)`, repeatable | unsorted |

```bash
curl "http://localhost:8080/api/v1/sections?page=0&size=10"
```

### `GET /api/v1/sections/{id}`

A single section definition by id. `404` if `id` is unknown.

```bash
curl "http://localhost:8080/api/v1/sections/44469"
```

---

## Speed/status history — `/api/v1/sections/{sectionId}/speed`

### `GET /api/v1/sections/{sectionId}/speed`

Speed/status measurement history for one section, newest first.

**Path parameters**

| Parameter | Description |
|---|---|
| `sectionId` | The section's id (matches `DefinitionOfSection.id`) |

**Query parameters**: same `Pageable` binding as above (`page`, `size`, `sort`).

```bash
curl "http://localhost:8080/api/v1/sections/44469/speed?size=50"
```

---

## Traffic messages — `/api/v1/traffic-messages`

### `GET /api/v1/traffic-messages`

Paginated list of traffic incidents/roadworks.

```bash
curl "http://localhost:8080/api/v1/traffic-messages?page=0&size=10"
```

### `GET /api/v1/traffic-messages/{id}`

A single traffic message by id. `404` if `id` is unknown.

```bash
curl "http://localhost:8080/api/v1/traffic-messages/123456"
```

---

## Machine-learning speed data — `/api/v1/sections/{id}/ml-speed-data`

### `GET /api/v1/sections/{id}/ml-speed-data`

Serves the same speed/status history as `/api/v1/sections/{sectionId}/speed`, but
re-shaped by the `SpeedOfSectionMLData` service into the numeric feature set consumed by
the traffic-speed ML model, instead of the raw entity fields.

**Path parameters**

| Parameter | Description |
|---|---|
| `id` | The section's id (matches `DefinitionOfSection.id`) |

**Query parameters**: same `Pageable` binding as the other list endpoints (`page`,
`size`, `sort`) — sorted newest-measurement-first by default.

```bash
curl "http://localhost:8080/api/v1/sections/44469/ml-speed-data"

# a smaller page, useful when spot-checking a specific section
curl "http://localhost:8080/api/v1/sections/44469/ml-speed-data?size=5"
```

**Response** — a JSON array, one object per measurement:

```json
[
  {
    "sectionId": 44469,
    "measureTime": "2026-09-13T13:44:00",
    "status": "freeflow",
    "statusEnum": 0,
    "speed": 70,
    "dayNr": 6,
    "daysUntilHoliday": 48,
    "holiday_type_regular_day": 1,
    "holiday_type_eve": 0,
    "holiday_type_holiday": 0,
    "freeflow_status": 1,
    "heavy_status": 0,
    "congested_status": 0,
    "imposible_status": 0,
    "minutesSincDaybreak": 464,
    "monthOfYear": 9,
    "holiday_is_nyar": 0,
    "holiday_is_jul": 0,
    "holiday_is_forsta_maj": 0,
    "holiday_is_nationaldagen": 0,
    "holiday_is_pask": 0,
    "holiday_is_kristihimmelsfard": 0,
    "holiday_is_pingst": 0,
    "holiday_is_midsommar": 0,
    "holiday_is_allahelgona": 1,
    "holiday_is_trettondag": 0
  }
]
```

### Field mappings

| Field | Type | Meaning |
|---|---|---|
| `sectionId` | number | Section id, copied from the measurement's composite key |
| `measureTime` | string (ISO local date-time) | When the measurement was taken |
| `status` | string | Raw status as reported by trafiken.nu (`freeflow`, `heavy`, `congested`, `impossible`) |
| `statusEnum` | number | `status` mapped to an integer — see below |
| `speed` | number \| `null` | Measured speed (km/h) as reported by trafiken.nu; `null` if not reported |
| `dayNr` | number | Day of week of `measureTime` — see below |
| `daysUntilHoliday` | number | Days from `measureTime`'s date until the next Swedish public holiday (see [Holiday calculation](#holiday-calculation)) |
| `holiday_type_regular_day`, `holiday_type_eve`, `holiday_type_holiday` | number (0/1) | One-hot encoding of whether `measureTime`'s date is a regular day, a holiday eve, or a holiday itself — exactly one is `1` — see below |
| `freeflow_status`, `heavy_status`, `congested_status`, `imposible_status` | number (0/1) | One-hot encoding of `statusEnum` — exactly one is `1` — see below |
| `minutesSincDaybreak` | number | Minutes elapsed since 06:00 on `measureTime`'s date, wrapping past midnight — see below |
| `monthOfYear` | number | Calendar month of `measureTime`, `1`–`12` |
| `holiday_is_nyar`, `holiday_is_jul`, `holiday_is_forsta_maj`, `holiday_is_nationaldagen`, `holiday_is_pask`, `holiday_is_kristihimmelsfard`, `holiday_is_pingst`, `holiday_is_midsommar`, `holiday_is_allahelgona`, `holiday_is_trettondag` | number (0/1) | One-hot encoding of which Swedish public holiday is next after `measureTime`'s date — exactly one is `1` — see below |

The one-hot fields exist so the ML model consumes plain numeric features directly,
without needing to embed a categorical/enum value itself.

#### `statusEnum` / `*_status`

| `status` | `statusEnum` | one-hot field set to `1` |
|---|---|---|
| `freeflow` | `0` | `freeflow_status` |
| `heavy` | `1` | `heavy_status` |
| `congested` | `2` | `congested_status` |
| `impossible` | `3` | `imposible_status` |

An unrecognized `status` value fails the request (`500`) rather than being silently
mapped — the four values above are the complete set observed from trafiken.nu.

#### `dayNr`

ISO day of week, zero-based starting on Monday:

| Day | `dayNr` |
|---|---|
| Monday | `0` |
| Tuesday | `1` |
| Wednesday | `2` |
| Thursday | `3` |
| Friday | `4` |
| Saturday | `5` |
| Sunday | `6` |

#### `holiday_type_*`

| One-hot field set to `1` | Meaning |
|---|---|
| `holiday_type_regular_day` | Regular day |
| `holiday_type_eve` | Eve — the day immediately before a Swedish public holiday |
| `holiday_type_holiday` | The date itself is a Swedish public holiday |

If a date is both (e.g. the day before a holiday that is itself preceded by another
holiday), `holiday_type_holiday` takes precedence over `holiday_type_eve`.

#### `minutesSincDaybreak`

Minutes elapsed since 06:00 on `measureTime`'s calendar day, ignoring seconds. A
measurement taken before 06:00 counts forward from *the previous day's* 06:00, so the
value always stays in `[0, 1439]`:

| `measureTime` (time part) | `minutesSincDaybreak` |
|---|---|
| `06:00` | `0` |
| `13:44` | `464` |
| `05:59` | `1439` (i.e. one minute before the next daybreak) |

#### `holiday_is_*`

A predefined, year-independent one-hot encoding of *which* Swedish public holiday
`daysUntilHoliday` counts down to — i.e. the same holiday `SwedishHolidays.nextHoliday()`
found for the given date. It is populated regardless of `holiday_type_*` (a regular day
still has exactly one `holiday_is_*` field set, for its upcoming holiday).

| One-hot field set to `1` | Holiday | internal `holidayNum` |
|---|---|---|
| `holiday_is_nyar` | Nyårsdagen (New Year's Day) | `1` |
| `holiday_is_jul` | Jul (Christmas) | `2` |
| `holiday_is_forsta_maj` | Första maj (May Day) | `3` |
| `holiday_is_nationaldagen` | Nationaldagen (National Day) | `4` |
| `holiday_is_pask` | Påsk (Easter) | `6` |
| `holiday_is_kristihimmelsfard` | Kristi himmelsfärdsdag (Ascension Day) | `7` |
| `holiday_is_pingst` | Pingstdagen (Whit Sunday) | `8` |
| `holiday_is_midsommar` | Midsommardagen (Midsummer Day) | `9` |
| `holiday_is_allahelgona` | Alla helgons dag (All Saints' Day) | `10` |
| `holiday_is_trettondag` | Trettondag jul (Epiphany) | `11` |

#### Holiday calculation

`daysUntilHoliday` and the `holiday_type_*` fields are both derived from
`SwedishHolidays` (see `src/main/java/com/tibell/trafficml/util/SwedishHolidays.java`),
which computes the fixed-date, Easter-based, and weekday-based Swedish public holidays:
Nyårsdagen, Trettondag jul, Första maj, Nationaldagen, Jul, Påsk, Kristi
himmelsfärdsdag, Pingstdagen, Midsommardagen, and Alla helgons dag.
