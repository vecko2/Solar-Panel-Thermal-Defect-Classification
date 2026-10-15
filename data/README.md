# Dataset

Download the original [InfraredSolarModules dataset](https://github.com/RaptorMaps/InfraredSolarModules) from Raptor Maps and extract it so this directory contains `module_metadata.json` and `images/*.jpg`. The notebook never downloads the data for you.

The metadata is a JSON object keyed by image ID. Each record has an `image_filepath` relative to this directory and an `anomaly_class`. The loader checks the class, path, image integrity, grayscale mode, and expected 24 × 40 pixel dimensions (width × height) before creating a split.

These JPEGs are visual thermal images, not calibrated temperature measurements. The metadata does not identify a plant, inspection session, or camera pass, so image level splits cannot rule out related images appearing across train, validation, and test. Exact pixel duplicates are kept together; visually similar images still require manual review.
