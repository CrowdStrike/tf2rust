# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] 2023-09-12

### Changed

- Update `tensorflow` to `2.13`
- Update generated template Rust code dependencies
- Update Rust edition to 2021
- Use `once_cell` instead of `lazy_static`

## [0.3.0] 2023-07-12

### Changed

- Modify package to make it a python wheel that is buildable with poetry
- Moved to tox and pytest

## [0.2.0] 2022-11-08

### Fixed

- Added support for Tensorflow > 2.5

## [0.1.0] 2022-10-10

- Initial release.

### Added

- Added changelog tracking.
- Added CI/CD tooling to run builds for the repo.
