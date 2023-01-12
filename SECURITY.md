[![Twitter URL](https://img.shields.io/twitter/url?label=Follow%20%40CrowdStrike&style=social&url=https%3A%2F%2Ftwitter.com%2FCrowdStrike)](https://twitter.com/CrowdStrike)

# Security Policy

This document outlines security policy and procedures for the CrowdStrike `tf2rust` project.

* [Supported versions](#supported-versions)
* [Reporting a potential security vulnerability](#reporting-a-potential-security-vulnerability)
* [Disclosure and Mitigation Process](#disclosure-and-mitigation-process)

## Supported versions

When discovered, we release security vulnerability patches for the most recent release at an accelerated cadence.

## Reporting a potential security vulnerability

We have multiple avenues to receive security-related vulnerability reports.

Please report suspected security vulnerabilities by:

* Submitting a [bug](https://github.com/CrowdStrike/tf2rust/issues/new?assignees=&labels=bug+%3Abug%3A&template=bug_report.md&title=%5B+BUG+%5D+...).
* Starting a new [discussion](https://github.com/CrowdStrike/tf2rust/discussions).
* Submitting a [pull request](https://github.com/CrowdStrike/tf2rust/pulls) to potentially resolve the issue. (New contributors: please review the content located [here](https://github.com/CrowdStrike/tf2rust/blob/main/CONTRIBUTING.md).)
* Sending an email to __dsci-oss@crowdstrike.com__.

## Disclosure and mitigation process

Upon receiving a security bug report, the issue will be assigned to one of the project maintainers. This person will coordinate the related fix and release
process, involving the following steps:

* Communicate with you to confirm we have received the report and provide you with a status update.

  * You should receive this message within 48 - 72 business hours.

* Confirmation of the issue and a determination of affected versions.
* An audit of the codebase to find any potentially similar problems.
* Preparation of patches for all releases still under maintenance.

  * These patches will be submitted as a separate pull request and contain a version update.
  * This pull request will be flagged as a security fix.
  * Once merged, and after post-merge unit testing has been completed, the patch will be immediately published to both PyPI repositories.

## Comments

If you have suggestions on how this process could be improved, please let us know by [starting a new discussion](https://github.com/CrowdStrike/tf2rust/discussions).
