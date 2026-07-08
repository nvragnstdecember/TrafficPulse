# Security Policy

TrafficPulse is a **research-stage** academic project. It is not a production
system and should not be deployed for real enforcement. Even so, we take
security and responsible disclosure seriously and appreciate reports.

## Reporting a vulnerability

**Please do not open a public issue, pull request, or discussion for a security
vulnerability**, and please do not disclose it publicly before it has been
addressed. Public disclosure puts users at risk before a fix exists.

Instead, report it privately through GitHub's built-in private reporting:

1. Go to the repository's **Security** tab.
2. Choose **Report a vulnerability** (GitHub Private Vulnerability Reporting).
3. Describe the issue with enough detail to reproduce it — affected file(s) or
   component, steps to reproduce, and impact.

This routes the report privately to the maintainer via GitHub, so no separate
email address is required. If private vulnerability reporting is not visible on
this repository, open a **minimal** issue that says only that you have a security
report to share and asks for a private channel — **without** including any
exploit details, proof-of-concept, or sensitive specifics in that public issue.

## Scope

This repository currently contains data contracts, governance/configuration, and
detector-independent reasoning components. It ships **no** trained models, **no**
datasets, and **no** network service. Reports about the code in this repository
(for example, unsafe parsing, deserialization, or dependency issues) are in
scope. Third-party frameworks, model weights, and datasets are governed by their
own upstream projects and should be reported to those projects.

## What to expect

As a research-stage project maintained on a best-effort basis, we do **not**
commit to a specific response-time or resolution SLA. We will make a reasonable
effort to review valid reports, confirm the issue, and address it, and to
coordinate on timing before any public disclosure.

Thank you for helping keep the project and its users safe.
