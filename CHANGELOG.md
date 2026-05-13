# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]


### Changed
- `picotech-adc-triggered`: new property `threshold` (good for photon counting, noise rejection)
- `picotech-adc-triggered`: new property `threshold_enabled` to switch on or off threshold
- `picotech-adc-triggered`: channel range enum changed for readability: "_50_mV" -> "mV_50"
- on backend: constants refactored to enums
- pyproject.toml updated; gui dependencies are separated
- nshots is now a property

### Fixed
- imp is deprecated; replace with importlib
- use valid avro enums for channel ranges
- added numpy vectorization to adc2mV conversion (drastic speedup)


## [2022.4.0]

### changed
- channel configuration no longer specifies sample/baseline baseline start/stop, or processing method.  Arbitrary processing to be done in processing scripts.
- daemon distinguishes between raw_channels, which are the native inputs of the device (e.g. A, B, C, D), and channels, which are the arbitrary outputs of the processing script.  Consequently, measure must be run once to establish stable/accurate channel parameters.
- removed `get_sample time` client message.  Use `get_mappings()["time"]` instead
- removed `get_measured_shots` client message.  `get_measured_samples` returns shots and samples data, so you can bin this array instead.

### fixed
- Shutdowns handle failed device close requests, which make client restarts more dependable.
- gui has responsive channel form for all channels.
- channel config `invert` now works

### Added
- initial release



[Unreleased]: https://gitlab.com/yaq/yaqd-picotech/-/compare/v2022.4.0...main
[2022.4.0]: https://gitlab.com/yaq/yaqd-picotech/-/tags/v2022.4.0
