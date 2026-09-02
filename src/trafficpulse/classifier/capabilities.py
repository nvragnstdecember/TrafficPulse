"""Backend capability checks: make a missing label loud instead of silent.

The problem this exists for is specific and real. The no-helmet rule exempts a rider whose
**predominant** observation is ``turban`` (``rules.no_helmet.exempt_riders``). That policy
is only as good as its evidence: it needs ``turban`` observations to actually arrive. Today
exactly one production backend can produce them -- the zero-shot classifier, via its
``turban`` prompt. A binary helmet/no_helmet backend (the P4-U5 winner) cannot emit the
label under any configuration.

Swap in such a backend and nothing raises. ``exempt_riders`` simply returns an empty set
forever, the exemption quietly becomes dead code, and turban-wearing riders start being
confirmed as no-helmet violations. **The failure is invisible at every layer**: the
classifier is behaving correctly, the adapter is mapping correctly, the reasoner is
applying its rule correctly, and the tests all pass. Only the outcome is wrong -- and it is
wrong in a direction that produces a systematic false-positive class against a religious
group, reversing the H8 real-footage fix.

That is a configuration error, not a runtime one, so it is caught where configuration is
turned into an engine -- the same place and the same way a missing classifier already fails
(``engine.rules.build_rules``). A backend states what it can say through
:attr:`~trafficpulse.classifier.interface.HelmetClassifier.supported_labels`; this module
compares that declaration against what a policy needs.

What this module deliberately does **not** do
---------------------------------------------
It does not choose a turban architecture. Composing a binary model with a turban-capable one,
deriving turban on its own evidence path, or retraining with a turban class are all open
options with different costs, and picking one is a decision for the project, not a side
effect of a safety check. This module only guarantees that whichever option is chosen, the
*unhandled* case cannot pass silently.

It also does not weaken the exemption, reinterpret ``turban``, or map it to anything. An
undeclared backend (``supported_labels is None``) is left alone: declaring capability is
opt-in, and absence of a declaration is not evidence of absence of the label.
"""

from __future__ import annotations

from .interface import HelmetClassifier

#: The native label the rule layer's turban exemption depends on. This is the *backend*
#: vocabulary string; mapping it onto ``HelmetState.TURBAN`` is the P4-U4 adapter's job
#: (``observations.helmet.DEFAULT_HELMET_LABEL_MAP``).
TURBAN_LABEL = "turban"


class ClassifierCapabilityError(Exception):
    """A configured policy depends on evidence the chosen backend cannot produce.

    Deliberately not a :class:`~trafficpulse.classifier.errors.HelmetClassifierError`: this
    is not a classification failure. Nothing went wrong at the seam -- the *combination* of
    backend and policy is unsound, which is a composition-time fact.
    """


def missing_labels(
    classifier: HelmetClassifier, required: frozenset[str]
) -> frozenset[str]:
    """Which ``required`` labels this backend has declared it cannot emit.

    Returns an empty set when the backend declares no vocabulary
    (``supported_labels is None``): undeclared means unknown, and unknown is not the same as
    incapable. Only an explicit declaration can prove a label is impossible.
    """

    declared = classifier.supported_labels
    if declared is None:
        return frozenset()
    return required - declared


def require_turban_capability(
    classifier: HelmetClassifier, *, acknowledged: bool = False
) -> frozenset[str]:
    """Fail unless the backend can emit ``turban``, or the operator has acknowledged it.

    Args:
        classifier: the backend about to be wired into a no-helmet rule.
        acknowledged: set only by an operator who has explicitly accepted running the
            no-helmet rule with a turban-blind backend. It does not make the consequence go
            away -- it records that someone chose it, so the choice is auditable rather than
            accidental. The returned set is still non-empty, so a caller can log or stamp it.

    Returns:
        The missing labels (empty when the backend is capable or undeclared).

    Raises:
        ClassifierCapabilityError: the backend has declared it cannot emit ``turban`` and
            the caller has not acknowledged that.
    """

    missing = missing_labels(classifier, frozenset({TURBAN_LABEL}))
    if missing and not acknowledged:
        raise ClassifierCapabilityError(
            f"{type(classifier).__name__} declares it cannot emit {sorted(missing)}, but the "
            "no_helmet rule's turban exemption needs it: without turban observations "
            "rules.no_helmet.exempt_riders can never exempt anyone, so a turban-wearing "
            "rider would be confirmed as a no-helmet violation. Use a turban-capable "
            "backend, or set acknowledge_turban_blind=True on the rule config to record "
            "that this consequence was accepted deliberately."
        )
    return missing
