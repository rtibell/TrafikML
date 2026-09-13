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
    "holidayNr": 0,
    "minutesSincDaybreak": 464,
    "monthOfYear": 9,
    "holidayNum": 14
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
| `holidayNr` | number | Whether `measureTime`'s date is a regular day, a holiday eve, or a holiday itself — see below |
| `minutesSincDaybreak` | number | Minutes elapsed since 06:00 on `measureTime`'s date, wrapping past midnight — see below |
| `monthOfYear` | number | Calendar month of `measureTime`, `1`–`12` |
| `holidayNum` | number | Which Swedish public holiday is next after `measureTime`'s date — see below |

#### `statusEnum`

| `status` | `statusEnum` |
|---|---|
| `freeflow` | `0` |
| `heavy` | `1` |
| `congested` | `2` |
| `impossible` | `3` |

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

#### `holidayNr`

| Value | Meaning |
|---|---|
| `0` | Regular day |
| `1` | Eve — the day immediately before a Swedish public holiday |
| `2` | The date itself is a Swedish public holiday |

If a date is both (e.g. the day before a holiday that is itself preceded by another
holiday), `2` takes precedence over `1`.

#### `minutesSincDaybreak`

Minutes elapsed since 06:00 on `measureTime`'s calendar day, ignoring seconds. A
measurement taken before 06:00 counts forward from *the previous day's* 06:00, so the
value always stays in `[0, 1439]`:

| `measureTime` (time part) | `minutesSincDaybreak` |
|---|---|
| `06:00` | `0` |
| `13:44` | `464` |
| `05:59` | `1439` (i.e. one minute before the next daybreak) |

#### `holidayNum`

A predefined, year-independent identifier for *which* Swedish public holiday
`daysUntilHoliday` counts down to — i.e. the same holiday `SwedishHolidays.nextHoliday()`
found for the given date. It is populated regardless of `holidayNr` (a regular day still
carries the number of its upcoming holiday, not `0`).

| `holidayNum` | Holiday |
|---|---|
| `1` | Nyårsdagen (New Year's Day) |
| `2` | Trettondedag jul (Epiphany) |
| `3` | Första maj (May Day) |
| `4` | Nationaldagen (National Day) |
| `5` | Juldagen (Christmas Day) |
| `6` | Annandag jul (Boxing Day) |
| `7` | Långfredagen (Good Friday) |
| `8` | Påskafton (Easter Eve) |
| `9` | Påskdagen (Easter Sunday) |
| `10` | Annandag påsk (Easter Monday) |
| `11` | Kristi himmelsfärdsdag (Ascension Day) |
| `12` | Pingstdagen (Whit Sunday) |
| `13` | Midsommardagen (Midsummer Day) |
| `14` | Alla helgons dag (All Saints' Day) |

#### Holiday calculation

`daysUntilHoliday` and `holidayNr` are both derived from `SwedishHolidays` (see
`src/main/java/com/tibell/trafficml/util/SwedishHolidays.java`), which computes the
fixed-date, Easter-based, and weekday-based Swedish public holidays: Nyårsdagen,
Trettondedag jul, Första maj, Nationaldagen, Juldagen, Annandag jul, Långfredagen,
Påskafton, Påskdagen, Annandag påsk, Kristi himmelsfärdsdag, Pingstdagen,
Midsommardagen, and Alla helgons dag.
