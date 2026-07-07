"""Simulated-penalty contract (simulation-only lifecycle state).

Data only: represents a *simulated* penalty. It implements no payment, legal
enforcement, notice delivery, or external integration. The model is
structurally simulation-only (architecture-review §21).
"""

from typing import Literal

from pydantic import AwareDatetime

from .enums import SimulatedPenaltyStatus
from .primitives import ContractModel, CurrencyCode, NonEmptyStr, NonNegativeFloat

# The mandated watermark text (architecture-review §21), locked as a Literal so
# it cannot be overridden on an instance.
DisclaimerText = Literal["SIMULATION - NOT A LEGAL NOTICE."]
SIMULATION_DISCLAIMER: DisclaimerText = "SIMULATION - NOT A LEGAL NOTICE."


class SimulatedAmount(ContractModel):
    """A simulated monetary amount (defaults to INR)."""

    value: NonNegativeFloat
    currency: CurrencyCode = "INR"


class SimulatedPenalty(ContractModel):
    """Simulated penalty state — never real enforcement.

    Two fields make the simulation structural rather than conventional:
    ``simulated`` is pinned to ``True`` (a ``Literal`` that cannot be set
    false), and ``disclaimer`` is locked to the mandated watermark string.
    Issuance presupposes an *approved* ``ReviewCase``; that human-approval gate
    is enforced by the Phase 2 workflow in code, not by this data contract.
    """

    penalty_id: NonEmptyStr
    review_case_id: NonEmptyStr
    status: SimulatedPenaltyStatus
    simulated: Literal[True] = True
    disclaimer: DisclaimerText = SIMULATION_DISCLAIMER
    amount: SimulatedAmount | None = None
    issued_at: AwareDatetime
    updated_at: AwareDatetime | None = None
