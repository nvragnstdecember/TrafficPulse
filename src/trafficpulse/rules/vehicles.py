"""Which detected classes can commit a road-vehicle movement violation.

One definition, shared by every rule that needs it. Stating it per rule would
create the drift this codebase warns about elsewhere (see
``app/capabilities.py``): two lists that agree today, disagree after the next
edit, and disagree *silently* -- the failure mode being a violation class that
quietly starts or stops applying to a road user.

Why a class filter is load-bearing
----------------------------------
``person`` is in the deployed label map because helmet and triple-riding
reasoning need riders, not because pedestrians commit movement violations. Every
geometry rule here reasons over a track's motion relative to configured road
geometry, and a pedestrian's motion is not that: people cross roads, stand in
them, and double back, so their tracks routinely oppose the traffic direction
and linger where vehicles may not. Without this filter those tracks are scored by
the same thresholds as a car -- observed directly during real-footage validation,
where a hi-vis traffic controller standing in a contraflow site produced
"against lane flow" readings at 142 deg and 178 deg and was held back from
confirming only by temporal persistence. That is a latent false positive resting
on a threshold, not on a decision.

``license_plate`` is excluded for the same structural reason: it is a region of
another detection, not an independently moving road user.

``bicycle`` is excluded, and this is the debatable one
------------------------------------------------------
A cyclist riding against traffic is a real offence in many jurisdictions, and a
cyclist crossing on red likewise -- but whether either is adjudicated, and under
which rules, varies by jurisdiction in a way this system has no basis to decide.
The project's existing red-light constant already excluded ``bicycle`` on those
grounds; keeping one set rather than two preserves that judgement instead of
quietly forking it. This is the member to revisit first if a deployment's
jurisdiction settles the question.
"""

from ..contracts.enums import ObjectClass

#: The classes a road-vehicle movement violation can be committed by. Consumed by
#: the wrong-way and red-light finalize strategies, both of which take it as a
#: parameter so a deployment can narrow or widen it without editing a rule.
VEHICLE_CLASSES: frozenset[ObjectClass] = frozenset(
    {
        ObjectClass.CAR,
        ObjectClass.MOTORCYCLE,
        ObjectClass.BUS,
        ObjectClass.TRUCK,
        ObjectClass.AUTO_RICKSHAW,
    }
)

__all__ = ["VEHICLE_CLASSES"]
