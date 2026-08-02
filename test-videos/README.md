# TrafficPulse Regression Dataset

The canonical collection of traffic footage used to verify that TrafficPulse still
works after every milestone. It exists so that "H12 didn't break H11" is something
we *check* rather than something we assert.

Used for: manual testing, regression runs, demonstrations, performance
measurement, and (eventually) published evaluation.

---

## Status, honestly

This is a **working dataset with a real gap**, and the gap is worth stating before
anything else.

| | |
|---|---|
| Clips registered | 9 |
| Clips downloaded | 2 (see [Fetching](#fetching)) |
| Clips from **fixed CCTV** | 0 |
| Clips from **India** | 4 registered |
| Clips with **frame-level ground truth** | 0 |
| Datasets identified but requiring manual acquisition | 8 |

**There is essentially no freely-redistributable Indian fixed-CCTV traffic video
on the open web.** The footage TrafficPulse is actually built for — overhead
Indian intersection CCTV, motorcycle-dense, annotated for helmet and rider
violations — exists, and it is very good, but every instance of it sits behind a
registration or a signed data agreement. The single most valuable action available
to this project is to obtain **AI City Challenge Track 5** (see
[Manual acquisition](#manual-acquisition)); it is Indian, fixed CCTV, 1080p, and
annotated per frame for exactly the two violations that run in production today.

What is here instead is real, correctly-licensed footage chosen to exercise
specific behaviours — congestion, occlusion, low resolution, night, contraflow —
plus a registry, a fetcher, an index, and an evaluation manifest that will accept
the good data the moment it arrives.

---

## Folder structure

```
test-videos/
├── README.md               this file
├── sources.yaml            the registry: every clip and dataset, with its licence
├── fetch.py                downloads the fetchable clips into place
├── build_index.py          probes the media, regenerates index.csv + metadata
├── index.csv               generated summary of the whole dataset
├── .gitignore              keeps media bytes out of git (see below)
│
├── no-helmet/              awaiting manual acquisition
├── triple-riding/          awaiting manual acquisition
├── wrong-way/              wrongway_001
├── illegal-stopping/       awaiting manual acquisition
├── red-light/              empty until H13 ships the reasoner
├── normal-traffic/         clean_001 … clean_004
├── edge-cases/
│   ├── night/              night_001
│   ├── rain/               empty — no correctly-licensed footage found
│   ├── congestion/         congestion_001, congestion_002
│   ├── occlusion/          covered by congestion_001/002; no dedicated clip yet
│   └── low-resolution/     lowres_001
└── evaluation/
    └── manifest.yaml       expected outputs, with the basis for each
```

Empty folders are **deliberate and meaningful**: they are the categories for which
no correctly-licensed footage was found, and they are where manually-acquired data
goes.

---

## Why the videos are not in the repository

`.gitignore` excludes every media extension. Three reasons, in order of weight:

1. **Licensing.** Some sources permit *use* but not *redistribution* — the Pexels
   and Pixabay licences both prohibit re-hosting their files, and several research
   datasets state no licence at all (which is not permission). Committing them
   would redistribute them.
2. **Git is the wrong store for video.** Binary blobs are never delta-compressed;
   a single 789 MB 4K clip is in every clone forever, including after deletion.
3. **Reproducibility is better served by the registry.** `sources.yaml` +
   `fetch.py` reconstructs the dataset from its sources, and records *where each
   clip came from* — which a committed blob does not.

What **is** committed: the registry, the scripts, `index.csv`, the per-video
`*.meta.yaml` sidecars, and the evaluation manifest. That is enough for anyone to
rebuild the dataset and to review its licensing without downloading a byte.

If you later decide the media should travel with the repo, use Git LFS — not plain
git — and only for the clips whose licence permits redistribution (`tier:
redistributable` in `sources.yaml`).

---

## Fetching

```bash
python test-videos/fetch.py --list       # what would be fetched
python test-videos/fetch.py              # everything except the large clips
python test-videos/fetch.py --large      # include clean_004 (789 MB) and night_001
python test-videos/fetch.py --only clean_002 wrongway_001
python test-videos/build_index.py        # re-measure and regenerate index.csv
```

`fetch.py` is idempotent — a clip already present is skipped — so re-running is
safe and resumes where a previous run stopped.

> **Wikimedia rate-limiting.** At the time of writing, `upload.wikimedia.org`
> returned HTTP 429 for most requests from this network, and 5 of the 7 default
> clips did not download. This is a throttle on the client's network, not a broken
> URL: the same script fetched `clean_001` and `congestion_002` successfully. The
> fetcher paces itself (`--delay`, default 4 s) and backs off exponentially on 429.
> If you are throttled, raise the delay (`--delay 30`) and fetch a few at a time
> with `--only`. Please do not remove the pacing — these are shared research
> mirrors.

### Transcoding

TrafficPulse accepts `.mp4 .avi .mkv .mov .webm .m4v`. Wikimedia serves a lot of
`.ogv`, which is **not** accepted — `index.csv` flags those as
`status: needs-transcode`. Convert before uploading:

```bash
ffmpeg -i clean_001.ogv -c:v libx264 -preset slow -crf 20 -an clean_001.mp4
```

Keep the original alongside it; the `.meta.yaml` sidecar describes the source file.

---

## Naming convention

`<prefix>_<NNN>.<ext>` — a three-digit sequence per prefix, and the **true**
container extension (never renamed to `.mp4` to make it look uniform; that would
misdescribe the file).

| Prefix | Category |
|---|---|
| `helmet_` | no-helmet |
| `triple_` | triple-riding |
| `wrongway_` | wrong-way |
| `stop_` | illegal-stopping |
| `redlight_` | red-light |
| `clean_` | normal-traffic |
| `night_` `rain_` `congestion_` `occlusion_` `lowres_` | edge-cases/* |

Numbers `001–099` are dataset-managed clips; reserve `100+` for clips you acquire
manually, so the two never collide.

---

## Metadata format

Every clip gets a `<filename>.meta.yaml` sidecar, generated by `build_index.py`:

```yaml
name: congestion_002.webm
source: Wikimedia Commons
title: A train stuck in traffic at a railway crossing in Raxaul, Bihar in 2018
license: CC BY-SA 4.0
attribution: See file page; CC BY-SA 4.0.
url: https://commons.wikimedia.org/wiki/File:...
duration: 13.29          # measured from the file, not advertised
resolution: 1920x1080    # measured
fps: 30.0                # measured
codec: vp9               # measured
camera_type: handheld
country: India
primary_violation: none
secondary_features: [india, motorcycle_heavy, congestion, level_crossing]
difficulty: hard
expected_behavior: >-
  ...
media_status: ok
measured: true           # false ⇒ not downloaded, technical fields are unknown
```

**Provider facts** (title, licence, url, what it shows) come from `sources.yaml`
and were read off the source page. **Technical facts** (duration, resolution, fps,
codec) are *measured* from the bytes by the project's own ingestion path, so they
are exactly what TrafficPulse will see. A clip that is not downloaded reports
`measured: false` and leaves those fields empty rather than copying the provider's
advertised numbers.

---

## Adding a video

1. Put the file in the right category folder with a conforming name.
2. Add an entry to `sources.yaml` under `videos:` — **including its licence and
   source URL**. An entry without a verified licence does not go in.
3. Choose the right `tier`:
   - `redistributable` — an explicit free licence (CC0/CC BY/CC BY-SA/PD).
   - `fetch_on_demand` — free to use, redistribution restricted (stock platforms).
   - `manual` — needs registration or a data agreement; `fetch.py` will not touch it.
4. Run `python test-videos/build_index.py`.
5. If — and only if — you can establish an expected outcome, add it to
   `evaluation/manifest.yaml` with its `basis`. See the manifest's header for what
   "establish" means here.

---

## Licensing summary

| Source | Licence | Redistributable? | In this dataset |
|---|---|---|---|
| Wikimedia Commons (CC0) | CC0 / public domain | Yes | `wrongway_001` |
| Wikimedia Commons (CC BY) | CC BY 3.0 / 4.0 | Yes, with attribution | `lowres_001`, `night_001` |
| Wikimedia Commons (CC BY-SA) | CC BY-SA 4.0 | Yes, attribution + share-alike | `clean_001–004`, `congestion_001/002` |
| Pexels | Pexels Licence | **No** — prohibits re-hosting | download-on-demand only |
| Pixabay | Pixabay Content Licence | **No** — prohibits re-hosting | download-on-demand only |
| AI City Challenge | Signed data agreement | **No** | manual, not committed |
| IITH Helmet 1 / 2 | **None stated**, citation required | **No** | manual, not committed |
| GRAM-RTM | "scientific research purposes", cite | **No** | manual, not committed |
| UA-DETRAC, BMD-45, UVH-26 | Research use / verify | **No** | manual, not committed |

Two points worth being explicit about:

- **No stated licence is not a free licence.** The IITH datasets state only a
  citation requirement. That is not permission to redistribute, and this dataset
  treats it accordingly.
- **CC BY-SA is share-alike.** If a derivative work (an annotated version, a
  montage) is distributed, it inherits the licence. Using a clip as a private test
  fixture does not trigger this; publishing a modified clip does.

Attribution for every committed-registry clip lives in its `.meta.yaml` sidecar and
in `sources.yaml`. Preserve it when you move files around.

---

## Manual acquisition

`sources.yaml` carries full instructions under `manual:`. In priority order:

1. **AI City Challenge Track 5** — *do this one first.* 200 videos, 20 s each,
   10 fps, 1920×1080, recorded in an Indian city, with per-frame boxes for
   motorcycle / driver-with-helmet / driver-without-helmet / three passenger
   positions each with helmet state. It is the only identified source that can turn
   `evaluation/manifest.yaml` from a plumbing check into an accuracy benchmark, and
   it matches TrafficPulse's two production rules exactly.
2. **IITH Helmet 1 & 2** — 3.5 hours of Hyderabad campus and city CCTV. Real fixed
   overhead viewpoints; no annotations, and no stated licence.
3. **GRAM-RTM Urban1** — a genuine fixed traffic-surveillance camera on a busy
   intersection with vehicle annotations. Spain, few motorcycles, but the *right
   camera geometry* for testing scene calibration and wrong-way.
4. **BMD-45**, **UA-DETRAC**, **UVH-26** (images only), **IITH Accident**.

---

## Core Benchmark Suite

The set to run after every H-phase milestone. It is specified at **18 slots**;
9 are filled today and the rest name exactly what fills them.

| # | Slot | Clip | Filled | What it proves |
|---|---|---|---|---|
| 1 | Clean / silence | `clean_002.webm` | registered | Zero events on orderly traffic — the false-positive floor |
| 2 | Clean / Indian | `clean_004.webm` | registered | Detector + tracker on dense Indian two-wheeler traffic |
| 3 | Clean / SE-Asian | `clean_001.ogv` | **downloaded** | Motorcycle-dense intersection; transcode path |
| 4 | Clean / roundabout | `clean_003.webm` | registered | Calibration refusal where no single legal direction exists |
| 5 | Wrong way | `wrongway_001.ogv` | registered | Wrong-way fires under a calibrated scene, and only then |
| 6 | Congestion | `congestion_001.webm` | registered | Tracker ID stability under occlusion; portrait aspect |
| 7 | Congestion / stationary | `congestion_002.webm` | **downloaded** | Illegal-stopping false-positive floor |
| 8 | Low resolution | `lowres_001.ogv` | registered | Graceful degradation, no fabricated events |
| 9 | Night | `night_001.webm` | registered | Detector robustness; no helmet claims from bad light |
| 10–14 | No helmet ×5 | `helmet_101–105` | **AI City T5** | Helmet accuracy against ground truth |
| 15–17 | Triple riding ×3 | `triple_101–103` | **AI City T5** | Rider-count accuracy against ground truth |
| 18 | Illegal stopping | `stop_101` | **IITH / GRAM** | Dwell detection in a drawn no-stopping zone |

Suggested cadence: run slots 1–9 after **every** milestone (they need no ground
truth and catch regressions in plumbing, calibration, and false-positive
behaviour); run the full 18 before any release or publication.

---

## Honesty

Three rules this dataset is built on. They are worth keeping.

1. **No fabricated expectations.** `evaluation/manifest.yaml` asserts a count only
   where it is *structural* (guaranteed by the system's design regardless of
   footage) or *annotated* (read from a provider's ground truth). Perception counts
   nobody has labelled are recorded as `established: false` with a null value — a
   TODO list, not assertions. A benchmark tuned to satisfy guesses is worse than no
   benchmark.
2. **Measured, not advertised.** Technical metadata comes from decoding the file.
3. **Verified licences only.** Every entry names a licence someone read on the
   source page, and "no licence stated" is recorded as exactly that.

---

## Future expansion

- **Obtain AI City Track 5.** Everything else on this list is secondary.
- **Label one clip by hand.** `clean_004` (4K Hyderabad) is the best candidate: a
  single careful pass gives the first real `no_helmet` / `triple_riding` targets and
  makes the manifest's `unestablished` entries into assertions.
- **Fill `edge-cases/rain/`.** No correctly-licensed rain footage was found;
  UA-DETRAC has weather-attributed sequences and is the likely source.
- **Red-light footage** becomes meaningful only once H13 ships the reasoner — and
  it needs clips where the signal head is visible *and* a signal schedule is known,
  which is a much narrower requirement than it looks.
- **Automate the comparison.** The manifest is machine-readable; a runner that
  processes each clip through the API and diffs against it would make regression
  checking a single command. Deliberately not built yet — it should follow the
  first annotated data, not precede it.
- **Consider a fixed-camera capture of your own.** A few hours of consented
  footage from one Indian junction, with a known signal cycle, would be worth more
  to this project than all the public datasets combined — and would be the only
  path to a red-light benchmark.
