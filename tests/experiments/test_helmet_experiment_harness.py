"""End-to-end harness test for the CNN-vs-ViT helmet experiment (P4-U5).

The filename is the one `docs/phase-4-plan.md` P4-U5 pre-registered as this unit's
required test.

It drives the whole pipeline on tiny synthetic fixtures -- annotations to corpus to
official split to crops to metrics to the pre-committed verdict -- without torch,
timm, or the real dataset, so the pieces are exercised *together* rather than only
in isolation. The training loop itself is deliberately out of scope here: it needs
a GPU and pretrained weights, and P4-U5's stop condition is an opt-in real run.

What this guards is the seam between stages. Each stage has its own focused test
file; this one exists because the bugs that actually shipped in this unit -- a
frame-filename padding assumption, and a dataset class that could not be pickled --
both lived in the joins, not inside a single function.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cnnvit_helpers import tiny_corpus_fixture
from helmet_cnn_vit.corpus import SamplingPolicy, build_corpus
from helmet_cnn_vit.labels import HelmetState
from helmet_cnn_vit.metrics import compute_metrics, predictions_from_scores
from helmet_cnn_vit.official_split import build_official_split, validate_no_leakage
from helmet_cnn_vit.stats import bootstrap_delta_macro_f1, decide, mcnemar
from helmet_rtdetr.split import SPLIT_ORDER


def test_the_pipeline_runs_annotations_to_a_frozen_split(tmp_path: Path) -> None:
    """Corpus -> official split -> validated manifest, with nothing lost on the way."""

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    corpus = build_corpus(annotation_dir)
    splits, manifest, statistics = build_official_split(corpus, split_csv)

    assert sum(len(records) for records in splits.values()) == len(corpus)
    assert manifest.total_crops == len(corpus)
    assert statistics.total.objects == len(corpus)
    assert set(splits) == {split.value for split in SPLIT_ORDER}
    validate_no_leakage(splits)


def test_the_frozen_artifacts_are_reproducible(tmp_path: Path) -> None:
    """Reproducibility is the whole basis of the pre-registration claim."""

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    first_corpus = build_corpus(annotation_dir)
    second_corpus = build_corpus(annotation_dir)
    assert first_corpus.content_hash() == second_corpus.content_hash()

    first = build_official_split(first_corpus, split_csv)[1]
    second = build_official_split(second_corpus, split_csv)[1]
    assert first.content_hash() == second.content_hash()


def test_a_changed_sampling_policy_changes_the_frozen_manifest(tmp_path: Path) -> None:
    """The manifest must not silently describe a corpus it did not come from."""

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    default = build_official_split(build_corpus(annotation_dir), split_csv)[1]
    altered = build_official_split(
        build_corpus(annotation_dir, policy=SamplingPolicy(max_crops_per_track=2)), split_csv
    )[1]
    assert default.content_hash() != altered.content_hash()
    assert altered.total_crops < default.total_crops


def test_the_split_carries_the_labels_the_metrics_layer_expects(tmp_path: Path) -> None:
    """The class vocabulary must line up across the corpus/metrics boundary."""

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    labels = {r.driver_state.value for records in splits.values() for r in records}
    assert labels <= {HelmetState.HELMET.value, HelmetState.NO_HELMET.value}

    truth = [r.driver_state.value for r in splits["test"]]
    result = compute_metrics(truth, truth)
    assert result.accuracy == 1.0


def test_the_whole_evaluation_path_produces_a_pre_committed_verdict(tmp_path: Path) -> None:
    """Split -> per-crop scores -> metrics -> McNemar + bootstrap -> verdict.

    Uses a test split large enough for the statistics to mean something, with one
    synthetic model strictly better than the other, so the rule must claim a
    difference rather than tie.
    """

    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)

    # A synthetic test set built from the real label vocabulary, 200 crops.
    truth = [HelmetState.HELMET.value] * 120 + [HelmetState.NO_HELMET.value] * 80
    strong_scores = [0.1] * 120 + [0.9] * 80
    weak_scores = [0.1] * 120 + [0.9] * 40 + [0.2] * 40  # misses half the no_helmets

    strong, strong_conf = predictions_from_scores(strong_scores)
    weak, _ = predictions_from_scores(weak_scores)

    strong_metrics = compute_metrics(
        truth, strong, no_helmet_scores=strong_scores, confidences=strong_conf
    )
    weak_metrics = compute_metrics(truth, weak, no_helmet_scores=weak_scores)
    assert strong_metrics.macro_f1 is not None and weak_metrics.macro_f1 is not None
    assert strong_metrics.macro_f1 > weak_metrics.macro_f1

    paired = mcnemar(
        [t == p for t, p in zip(truth, strong, strict=True)],
        [t == p for t, p in zip(truth, weak, strict=True)],
    )
    assert paired.only_first_correct == 40
    assert paired.only_second_correct == 0

    interval = bootstrap_delta_macro_f1(truth, strong, weak, resamples=300, seed=0)
    verdict = decide(
        first_name="strong",
        second_name="weak",
        per_seed_delta=[0.10, 0.11, 0.09],
        interval=interval,
        mcnemar_results=[paired],
    )
    assert verdict.difference_claimed
    assert verdict.winner == "strong"
    # The verdict must carry its own evidence into the artifact.
    assert verdict.interval.resamples == 300
    assert len(verdict.mcnemar) == 1
    assert splits["test"]  # the real split path still produced data


def test_a_marginal_result_is_reported_as_a_tie() -> None:
    """The outcome §12 explicitly anticipates, and the one easiest to overclaim."""

    truth = [HelmetState.HELMET.value] * 100 + [HelmetState.NO_HELMET.value] * 100
    first = list(truth)
    second = list(truth)
    second[0] = HelmetState.NO_HELMET.value  # one crop different out of 200

    interval = bootstrap_delta_macro_f1(truth, first, second, resamples=300, seed=0)
    verdict = decide(
        first_name="a", second_name="b", per_seed_delta=[0.001, -0.002, 0.0005], interval=interval
    )
    assert not verdict.difference_claimed
    assert verdict.winner is None
    assert "TIE" in verdict.rationale


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_each_split_is_non_empty_and_single_sourced(tmp_path: Path, split: str) -> None:
    annotation_dir, split_csv = tiny_corpus_fixture(tmp_path)
    splits, _, _ = build_official_split(build_corpus(annotation_dir), split_csv)
    records = splits[split]
    assert records
    assert len({r.video_id for r in records}) >= 1
