# MIPD package-level post-DD FOC assessment - V9.11

V9.11 uses vessel-specific Huber machine learning to estimate package-level post-dry-dock main-engine FOC saving from noon reports. It compares unseen-data prediction performance against a public cubic-speed benchmark, `FOC = a x STW^3`, rather than a company-specific method. The benchmark coefficient is fitted using pre-DD data only and the benchmark is presented as a rule of thumb, not an ISO-prescribed FOC-saving method.

## Run on Windows

Extract the ZIP into a new folder. Open a terminal in the folder containing `app.py` and `requirements.txt`, then run:

```text
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Replace the complete folder rather than only `app.py`; the model, processing, audit, report and export modules changed together.

## V9.11 assessment rules

- Main-engine fuel, LOG distance and propelling hours must cover the same report interval.
- At least 18 propelling hours.
- Beaufort 4 or below.
- A 5-35 kn bound is used only to reject physically implausible steady-sea records.
- The former fixed 13-25 kn operating filter is not used.
- Actual speed/loading comparability is learned from valid pre-DD reports using local density in log STW and log displacement.
- The same confirmed route and operating leg are preferred.
- If strict evidence does not reach 10 reports and 40% comparable fuel coverage, the app tests an expanded same-leg cross-route comparison.
- Expanded cross-route results are capped at `Preliminary`.
- Gross FOC anomalies are visibly flagged but are not silently deleted. The app retrains without flagged reports as a sensitivity check.
- The selected Huber model is compared with a public cubic-speed benchmark on the same unseen pre-DD reports and comparable post-DD reports.
- A calculated percentage is classified as Inconclusive when the pre-DD model fails the declared MAPE, bias or learned-speed-exponent validation gates.
- The conclusion is classified as Unstable when reasonable alternative analyses change the direction between saving and higher FOC.
- All unsupported or rejected reports remain in the audit and fuel-coverage denominator where applicable.

## App structure

1. **Setup** - check the uploaded workbook, confirm operational evidence, review the declared rules and run the assessment.
2. **Result summary** - presents the conclusion, selected reports, package-level calculation and the checks affecting use.
3. **Supporting analysis** - provides model validation, same-route/cross-route results, the STW/displacement diagnostic, report audit, methodology and downloads.

The built-in synthetic demonstration contains generated data with an approximately 10% post-DD reduction. It explains the workflow and is not vessel evidence.

## Interpretation

Positive saving means reported post-DD fuel was lower than fuel expected from the selected pre-DD Huber relationship over the same comparable reports and propelling hours. Negative saving means reported fuel was higher.

The result is an observational, steady-at-sea, package-level main-engine FOC estimate. It is not total voyage fuel, proof that dry docking alone caused the difference, certified performance verification, an ISO 19030 calculation, or saving attributed to an individual work item. The overall adjusted-baseline concept is consistent with general measurement-and-verification principles, but the prototype does not claim ISO 50015 certification or conformance.

## Local tests

```text
py -m unittest discover -p "test*.py" -v
```

Runtime data remains in the local Streamlit session. The app does not send the uploaded workbook to an external service.
