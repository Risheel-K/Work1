import numpy as np
from datetime import datetime, timedelta, timezone
from sgp4.api import Satrec
from astropy.time import Time
from astropy.coordinates import (
    TEME, ITRS,
    CartesianRepresentation, CartesianDifferential,
    EarthLocation, get_sun,
)
import astropy.units as u
import warnings
warnings.filterwarnings("ignore")

R_EARTH_KM = 6378.137

OBJECT_COSPAR = "2025-052P"
OBJECT_NORAD  = 63223
TLE_LINE1 = "1 63223U 25052P   25244.59601767  .00010814  00000-0  51235-3 0  9991"
TLE_LINE2 = "2 63223  97.4217 137.0451 0006365  74.2830 285.9107 15.19475170 25990"

TRACKER = {
    "name"              : "Svalbard Ground Station (SvalSat vicinity)",
    "geodetic_lat_deg"  : 78.9066,
    "geodetic_lon_deg"  : 11.88916,
    "alt_msl_m"         : 380.0,
    "for_half_angle_deg": 70.0,
    "min_elevation_deg" : 20.0,
}

EPOCH_START_UTC  = datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
DURATION_DAYS    = 1.0
DT_SECONDS       = 10.0
SUN_NIGHT_EL_DEG = 0.0


def ccsds_doy(dt: datetime) -> str:
    return dt.strftime("%Y-%jT%H:%M:%S.000 UTC")


def build_time_grid(t0, dur_days, dt_sec):
    n       = int(dur_days * 86400 / dt_sec) + 1
    dt_list = [t0 + timedelta(seconds=i * dt_sec) for i in range(n)]
    t_ast   = Time(dt_list, format="datetime", scale="utc")
    return dt_list, t_ast


def propagate_sgp4(sat, times_ast):
    jd      = times_ast.jd
    jd_i    = np.floor(jd)
    jd_frac = jd - jd_i
    err, pos, vel = sat.sgp4_array(jd_i, jd_frac)
    return pos, vel, err.astype(int)


def teme_to_itrs(pos_teme, times_ast):
    zeros = np.zeros(len(times_ast))
    r     = CartesianRepresentation(
                pos_teme[:, 0] * u.km,
                pos_teme[:, 1] * u.km,
                pos_teme[:, 2] * u.km)
    dv    = CartesianDifferential(zeros * u.km/u.s,
                                  zeros * u.km/u.s,
                                  zeros * u.km/u.s)
    frame = TEME(r.with_differentials(dv), obstime=times_ast)
    itrs  = frame.transform_to(ITRS(obstime=times_ast))
    return itrs.cartesian.xyz.to(u.km).value.T


def sun_itrs_batch(times_ast):
    sun_gcrs = get_sun(times_ast)
    sun_itrs = sun_gcrs.transform_to(ITRS(obstime=times_ast))
    return sun_itrs.cartesian.xyz.to(u.km).value.T


def geodetic_to_itrs(lat_deg, lon_deg, alt_m):
    loc = EarthLocation(lat=lat_deg * u.deg,
                        lon=lon_deg * u.deg,
                        height=alt_m * u.m)
    return np.array([loc.x.to(u.km).value,
                     loc.y.to(u.km).value,
                     loc.z.to(u.km).value])


def elevation_range_batch(sat_itrs, gs_itrs, lat_deg, lon_deg):
    phi   = np.radians(lat_deg)
    lam   = np.radians(lon_deg)
    Z_hat = np.array([np.cos(phi) * np.cos(lam),
                      np.cos(phi) * np.sin(lam),
                      np.sin(phi)])
    rho      = sat_itrs - gs_itrs
    rho_norm = np.linalg.norm(rho, axis=1)
    rho_Z    = rho @ Z_hat
    el_deg   = np.degrees(np.arcsin(np.clip(rho_Z / rho_norm, -1.0, 1.0)))
    return el_deg, rho_norm


def is_sunlit_batch(sat_itrs, sun_itrs):
    sun_norm = np.linalg.norm(sun_itrs, axis=1, keepdims=True)
    s_hat    = sun_itrs / sun_norm
    proj     = np.einsum("ij,ij->i", sat_itrs, s_hat)
    perp     = sat_itrs - proj[:, np.newaxis] * s_hat
    perp_d   = np.linalg.norm(perp, axis=1)
    return (proj > 0.0) | (perp_d > R_EARTH_KM)


def find_event_windows(mask, times_dt, elevations, ranges):
    events, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            sl = slice(i, j)
            pk = i + int(np.argmax(elevations[sl]))
            events.append({
                "start"       : times_dt[i],
                "end"         : times_dt[j - 1],
                "i0"          : i,
                "i1"          : j - 1,
                "duration_s"  : (times_dt[j - 1] - times_dt[i]).total_seconds(),
                "peak_el_deg" : float(elevations[pk]),
                "peak_time"   : times_dt[pk],
                "min_range_km": float(np.min(ranges[sl])),
                "max_range_km": float(np.max(ranges[sl])),
            })
            i = j
        else:
            i += 1
    return events


def print_event_table(events, section_label):
    if not events:
        print("    ⊘  No events detected within the analysis window.\n")
        return

    print(f"    Total events : {len(events)}\n")
    W = [4, 29, 29, 9, 9, 12]
    H = ["#", "AOS (CCSDS DOY / UTC)", "LOS (CCSDS DOY / UTC)",
         "Dur", "El_max", "RangeMin km"]
    row_fmt = "    " + "  ".join(f"{{:<{w}}}" for w in W)
    print(row_fmt.format(*H))
    print("    " + "─" * (sum(W) + 2 * (len(W) - 1)))

    for idx, ev in enumerate(events, 1):
        d_s = ev["duration_s"]
        row = row_fmt.format(
            str(idx),
            ccsds_doy(ev["start"]),
            ccsds_doy(ev["end"]),
            f"{int(d_s//60)}m{int(d_s%60):02d}s",
            f"{ev['peak_el_deg']:+.2f}\u00b0",
            f"{ev['min_range_km']:,.1f}",
        )
        print(row)
        print(f"       El_max epoch : {ccsds_doy(ev['peak_time'])}")
    print()


def main():
    SEP = "═" * 72

    print(f"\n╔{SEP}╗")
    print(f"║  SPACE SURVEILLANCE — CROSSING & VISIBILITY ANALYSIS              ║")
    print(f"║  Ground-Based Sensor | SGP4 Propagator | CCSDS 502.0-B-3         ║")
    print(f"╚{SEP}╝\n")

    gs     = TRACKER
    MIN_EL = gs["min_elevation_deg"]

    print(f"  Object   : COSPAR {OBJECT_COSPAR} | NORAD {OBJECT_NORAD}")
    print(f"             SSO ~550 km | i = 97.42° | BSTAR = 5.12×10⁻⁴ m⁻¹")
    print(f"  Tracker  : {gs['name']}")
    print(f"             φ = {gs['geodetic_lat_deg']}°N  "
          f"λ = {gs['geodetic_lon_deg']}°E  "
          f"h = {gs['alt_msl_m']} m MSL  (WGS-84)")
    print(f"  FOR      : {gs['for_half_angle_deg']}° half-angle from zenith "
          f"→ Min elevation = {MIN_EL}°")
    print(f"  Window   : {ccsds_doy(EPOCH_START_UTC)} + {DURATION_DAYS:.0f} day "
          f"| Δt = {DT_SECONDS:.0f} s")
    print(f"  Twilight : Sun elevation < {SUN_NIGHT_EL_DEG}° (geometric night)\n")

    times_dt, times_ast = build_time_grid(EPOCH_START_UTC, DURATION_DAYS, DT_SECONDS)
    N = len(times_dt)

    print(f"  [1/4] SGP4 propagation ({N} epochs) ...", end="  ", flush=True)
    sat = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)
    pos_teme, _, err_codes = propagate_sgp4(sat, times_ast)
    n_err = int(np.sum(err_codes != 0))
    print(f"OK  ({n_err} SGP4 error epoch{'s' if n_err != 1 else ''})")

    if n_err > 0:
        print(f"  ⚠  SGP4 error codes: {np.unique(err_codes[err_codes != 0])}")
        print("     Affected epochs excluded from analysis.")

    print("  [2/4] TEME → ITRS transform (IAU-76/FK5) ...", end="  ", flush=True)
    sat_itrs = teme_to_itrs(pos_teme, times_ast)
    print("OK")

    print("  [3/4] Solar ephemeris (astropy / JPL) ...", end="  ", flush=True)
    sun_itrs_arr = sun_itrs_batch(times_ast)
    print("OK")

    print("  [4/4] Elevation / shadow / night computation ...", end="  ", flush=True)
    gs_itrs = geodetic_to_itrs(gs["geodetic_lat_deg"],
                                gs["geodetic_lon_deg"],
                                gs["alt_msl_m"])

    el_sat,  rng_sat = elevation_range_batch(sat_itrs, gs_itrs,
                                              gs["geodetic_lat_deg"],
                                              gs["geodetic_lon_deg"])
    el_sun,  _       = elevation_range_batch(sun_itrs_arr, gs_itrs,
                                              gs["geodetic_lat_deg"],
                                              gs["geodetic_lon_deg"])
    sat_lit          = is_sunlit_batch(sat_itrs, sun_itrs_arr)

    if n_err > 0:
        bad = err_codes != 0
        el_sat[bad]  = -90.0
        sat_lit[bad] = False

    station_night = el_sun < SUN_NIGHT_EL_DEG
    print("OK\n")

    crossing_mask   = el_sat >= MIN_EL
    crossing_events = find_event_windows(crossing_mask, times_dt, el_sat, rng_sat)

    print(f"  {'─'*70}")
    print(f"  SECTION 1 — CROSSING EVENTS")
    print(f"  Definition : Object geometrically inside FOR (El ≥ {MIN_EL}°)")
    print(f"  Note       : Crossing ≠ detection; detection conditions in §2")
    print(f"  {'─'*70}")
    print_event_table(crossing_events, "crossing")

    vis_mask   = crossing_mask & sat_lit & station_night
    vis_events = find_event_windows(vis_mask, times_dt, el_sat, rng_sat)

    print(f"  {'─'*70}")
    print(f"  SECTION 2 — VISIBILITY / DETECTION EVENTS")
    print(f"  Conditions : (a) El ≥ {MIN_EL}°  ∧  (b) Object Sunlit  "
          f"∧  (c) Station Night (Sun El < {SUN_NIGHT_EL_DEG}°)")
    print(f"  {'─'*70}")
    print_event_table(vis_events, "visibility")

    pct = lambda m: m.sum() / N * 100
    print(f"  {'─'*70}")
    print(f"  ANALYSIS SUMMARY")
    print(f"  {'─'*70}")
    print(f"    Propagation epochs              : {N:,}")
    print(f"    SGP4 error epochs               : {n_err}")
    print(f"    Crossing events                 : {len(crossing_events)}")
    print(f"    Visibility events               : {len(vis_events)}")
    print(f"    Time fraction inside FOR        : {pct(crossing_mask):.2f} %")
    print(f"    Time fraction object sunlit     : {pct(sat_lit):.2f} %")
    print(f"    Time fraction station night     : {pct(station_night):.2f} %")
    print(f"    Time fraction all conds. met    : {pct(vis_mask):.2f} %")
    print(f"  {'═'*70}\n")


if __name__ == "__main__":
    main()
