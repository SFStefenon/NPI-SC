# Dataset Description

## Overview

The dataset contains historical electricity distribution network interruption records and meteorological observations from Lages, Santa Catarina, southern Brazil, covering the period from 2016 to 2025. The electrical data originate from Centrais Elétricas de Santa Catarina (CELESC Distribuição), while meteorological measurements are obtained from the Brazilian National Institute of Meteorology (INMET).

The dataset integrates information on electricity supply interruptions, network reliability indicators, precipitation, atmospheric pressure, temperature, humidity, solar radiation, and wind conditions. These variables characterize the operational performance of the electricity distribution network under different environmental conditions.

## Electrical Distribution Data

The electrical records were obtained from CELESC's operational information systems, which maintain information about planned and unplanned electricity supply interruptions.

The dataset includes three electrical variables:

- **QTDE UC EQPTO INTERROMPIDA:** Number of consumer units affected by electricity supply interruptions. This variable represents affected consumer units rather than the number of individual fault events.
- **DEC (Equivalent Duration of Interruption per Consumer Unit):** Reliability indicator describing the average duration of electricity supply interruptions experienced by consumers.
- **FEC (Equivalent Frequency of Interruption per Consumer Unit):** Reliability indicator describing the average frequency of electricity supply interruptions experienced by consumers.

## Meteorological Data

Meteorological observations were collected from INMET weather station records associated with Lages. The original measurements have an hourly resolution and include the following environmental variables:

- **Precipitation:** Total accumulated rainfall, expressed in millimeters (mm).
- **Atmospheric pressure:** Mean, maximum, and minimum atmospheric pressure, expressed in millibars (mb).
- **Air temperature:** Mean, maximum, and minimum air temperature, expressed in degrees Celsius (°C).
- **Dew point temperature:** Mean, maximum, and minimum dew point temperature, expressed in degrees Celsius (°C).
- **Relative humidity:** Mean, maximum, and minimum atmospheric relative humidity, expressed as percentages (%).
- **Wind speed:** Mean and maximum wind speed, expressed in meters per second (m/s).
- **Wind gusts:** Maximum recorded wind gust speed, expressed in meters per second (m/s).
- **Wind direction:** Mean wind direction, expressed in degrees (0–360°).
- **Solar radiation:** Total accumulated global solar radiation, expressed in kJ/m².

The meteorological records also include observation-coverage information, identifying the number of registered measurements and the availability of valid precipitation and solar radiation measurements.

## Data Organization and Temporal Resolution

The electrical interruption records and hourly meteorological observations are consolidated into a daily dataset. Each row represents one calendar day, allowing electrical reliability measurements and environmental conditions to be examined on a common temporal scale.

The daily number of interrupted consumer units is calculated by summing the corresponding interruption records. The DEC and FEC values are aggregated using their daily arithmetic means. Precipitation and solar radiation are accumulated over each day, while the remaining meteorological variables are summarized using appropriate mean, minimum, or maximum statistics. Wind direction is aggregated using a circular mean.

Meteorological timestamps are converted from UTC to the local time zone of Lages before daily aggregation. The final dataset includes only the period shared by the electrical and meteorological records.

## Data Quality and Missing Values

The preprocessing procedure removes duplicate hourly meteorological timestamps, converts invalid numerical entries into missing values, and excludes physically invalid observations, including negative precipitation, negative wind speeds, and relative humidity outside the range of 0–100%.

Missing meteorological measurements are retained as missing values rather than replaced with zeros. Observation-count columns provide information about the completeness of the meteorological records. Electrical variables are assigned zero for dates without corresponding records during daily reindexing.

The resulting dataset is stored as `DEC_INMET_day.csv`, containing **24 numerical variables and a daily date index**. It combines electricity distribution performance indicators with environmental measurements across multiple years, providing a historical record of network interruptions and meteorological variability.
