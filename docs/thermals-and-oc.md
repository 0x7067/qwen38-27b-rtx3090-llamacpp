# Thermals, fan control, and overclocking on the RTX 3090

This rig runs GDDR6X, which throttles/errors around 110°C junction and whose
*junction* temperature `nvidia-smi` cannot report at all on GeForce cards
(`temperature.memory` is `N/A`). Everything here was measured with a
BAR-register memory-junction reader
([gputemps](https://github.com/ThomasBaruzier/gputemps)-style direct read),
not `nvidia-smi`.

## The finding that matters more than any overclock: the fan curve was blind to memory temperature

Stock behavior: both fans sit at **0%** (full stop) under sustained decode,
because the shipped fan curve follows *core* temperature. The core never
leaves 55–65°C on this workload, so the fans never spin up — while GDDR6X
junction quietly climbs to **98°C**, 2°C below the throttle/error region.

Forcing the fans to 100% (no overclock, same workload, 10-minute soak):

| | fans auto (0%) | fans 100% | change |
|---|---:|---:|---:|
| junction max | 98°C | **92°C** | **−6°C** |
| junction steady state | 96–98°C | 90–92°C | −6°C |
| decode t/s (mean of 12) | 103.87 | 104.69 | +0.79% (noise) |

**6°C of thermal margin for a throughput cost within measurement noise.**
This is the single highest-value finding in the whole thermal/OC campaign,
and it needed no overclocking at all — just correcting a fan curve that was
watching the wrong sensor.

## NVML offset convention (resolved empirically)

`nvidia-smi`/NVML's memory VF offset units are **2:1**: an NVML offset of
`+500` moves `clocks.mem` by `+250 MHz` (measured: stock 9501 MHz →
`+500` gives 9751 MHz, which is also the card's P0 ceiling — the first 500
units of offset just buy back the P2-vs-P0 clock cap, not additional headroom
past P0).

## Memory offset ladder (with fans forced, GDDR6X-junction gated)

| NVML offset | `clocks.mem` | tg64@32k Δ | junction max | verdict |
|---:|---:|---:|---:|---|
| 0 (stock) | 9501 MHz | — | 98°C (auto fan) / 92°C (100% fan) | baseline |
| +1000 | 10001 MHz | **+2.47%** | **94°C** (100% fan, 10-min soak) | **shipped** |
| +1500 | 10251 MHz | +3.41% | 86°C (short bench, not soaked) | untested sustained |
| +2000 (+ power limit 390W) | 10501 MHz | +5.90% decode / +3.78% prefill | **100°C — aborted at 202s (auto fan) / 552s (100% fan)** | **not shippable** |

Output was bit-identical (sha256, and KLD 0.00000) at every offset tested —
this is a clock/thermal question, not a numerics question.

`+2000` is electrically flawless but thermally unsustainable even at maximum
airflow, because it was evaluated together with a 390 W power-limit bump
(+39 W over stock, confirmed via continuous telemetry): the card saturates
whatever power limit it's given (351 W actual at a 370 W cap, 390 W actual at
a 390 W cap), so `+2000/390W` conflates two separate thermal loads. The `+1000`
arm isolates the memory-clock lever alone at the stock 370 W power limit.

## Shipped configuration

```bash
# 1. Force fans off the (memory-blind) core-temperature curve — the essential part
nvmlDeviceSetFanControlPolicy(MANUAL) + nvmlDeviceSetFanSpeed_v2(100)
# 2. Moderate memory offset: +1000 NVML units = +500 MHz -> clocks.mem 10001
nvmlDeviceSetMemClockOffset(+1000)
# power limit left at stock 370W; no core-clock lock (locking core clock
# measured *below* unlocked boost on this card — ruled out)
```

Net effect: **+2.47% decode throughput, plus a thermal margin improvement**
(98°C → 94°C junction under sustained load) rather than a thermal cost — the
fan-curve fix buys more headroom than the memory offset spends. `+1500`
and above were not adopted: the ladder ends at the *test's* ceiling, not a
regression, so a moderate, soak-verified rung was preferred over
extrapolating past what was actually measured under sustained load.

## The kill-switch incident this made visible

A same-day thermal kill-switch (`vram-thermal-guard`, a `pkill -9
llama-server` watchdog we installed) was configured with its trip point
(`VRAM_TRIP=98°C`) set from this same soak data — but at exactly the
temperature a *stock, unmodified* run reaches once the fan-curve dead zone
lets the fans idle at 0%. That is a kill-switch armed with zero margin above
normal operation, and it fired repeatedly in production the same day it was
installed, misread at the time as an engine crash (see
`docs/dflash2-findings.md`'s neighbor issue and the fan-duty dead-zone note
below for the mechanism). Fix: raise the trip point above the stock-workload
plateau (100–102°C) and/or fix the fan curve's dead zone so the plateau
itself drops.

### Related: commanded fan duty vs actual RPM has a dead zone

A duty-vs-RPM sweep on this card (`SetFanControlPolicy(MANUAL)` +
`SetFanSpeed_v2(p)`, read back after settling) found commanded duty and
actual spin are not linear at the low end:

| commanded duty | actual RPM |
|---:|---:|
| 30% | 0% |
| 40% | 0% |
| 50% | 50% |
| 60% | 60% |
| 100% | 99% |

A fan-curve minimum floor set inside this dead zone (e.g. 30%) silently
produces stopped fans, which is exactly the mechanism behind the "0% fans
at 98°C junction" finding above. Any junction-keyed fan daemon on this class
of card needs its floor probed and clamped above the dead zone, not assumed
linear from the commanded value.
