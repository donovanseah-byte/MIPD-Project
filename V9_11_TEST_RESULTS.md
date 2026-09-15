# V9.11 verification summary

## Automated checks

- 46 unit and integration tests passed, including direct coverage of all five conclusion paths.
- Python compilation completed without an error.
- Streamlit setup opened without an exception.
- The built-in synthetic demonstration completed without an exception.
- The Streamlit server started without an application exception.
- Synthetic demonstration ML result was 9.95%; the public cubic-speed benchmark result was 9.77%.
- Synthetic demonstration sensitivity range was 9.93%-9.98%.
- No company-specific benchmark name, exponent or result field remained in the source, interface, exports or methodology.

## Eight-workbook regression check

All eight uploaded workbooks completed successfully using their confirmed dry-dock dates. The primary Huber estimates, supported-report counts and selected same-route/cross-route comparison bases were unchanged because the primary ML calculation was not modified.

MV Pioneer changed from Preliminary to Inconclusive because its chronological Huber MAPE was approximately 125.6%, above the declared 12% validation gate. The other seven classifications were unchanged.

The selected Huber model had lower or equal unseen-data MAPE than the public cubic-speed benchmark in four of the eight workbooks. The cubic benchmark had lower MAPE in the other four. This is reported honestly as benchmark evidence; it does not replace the Huber result automatically.

One workbook retained its previously identified gross-FOC review flag. The row remained in the primary result and the anomaly-excluded sensitivity remained visible.

## Presentation checks

- The main page contains four concise sections: conclusion, report selection, package-level calculation and checks affecting use.
- The conclusion is no longer repeated as a separate metric.
- The methods table appears only in Method and downloads.
- The main page uses one calculation visual instead of two visuals showing the same expected and reported fuel totals.
- The STW/displacement scatterplot is absent from the main page and available only in a collapsed technical diagnostic.
- The selected same-route or cross-route comparison states its data floor and selection reason.
- The ISO 19030 and ISO 50015 scope statements are collapsed under `Method scope and standards reference`.
- The public cubic-speed benchmark is identified in the methods table, model-reliability section, exports and printable report.
- Primary results retain two-decimal display precision.
