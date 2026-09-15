# V9.11 evidence classification and interface wording update

V9.11 prevents a poorly validated pre-DD model from receiving a Preliminary conclusion merely because the basic report-count and fuel-coverage floor was met. MAPE above 12%, absolute bias above 8%, unavailable validation or a learned speed exponent outside 1.5-4.5 now results in an Inconclusive classification.

The result page has also been shortened. Repeated conclusion, comparison-method and expected-versus-reported-fuel displays were removed or consolidated. Titles now distinguish the result summary, supporting analysis, comparable operating conditions, prediction validation and sensitivity-direction checks.

## Confidentiality change

- Removed the company name and company-specific benchmark formula from all user-facing text.
- Removed the company benchmark from validation, post-DD comparison, exports and printable reports.
- Removed company-specific benchmark identifiers from the calculation code.
- Added a public-source explanation and explicit non-conformance boundaries for ISO 19030 and ISO 50015.

## Public comparison method

- Fits `a` from valid pre-DD reports using least squares with a fixed exponent of 3.
- Compares Huber and cubic-speed MAPE, bias and RMSE on the same unseen pre-DD reports.
- Calculates the cubic-speed post-DD saving on the same comparable reports and propelling hours as the ML estimate.
- Labels the method as a public physics-based rule of thumb, not an ISO-prescribed saving method.

## Unchanged primary method

The selected Huber model, eligibility filters, strict/cross-route support hierarchy, post-DD report selection and propelling-hour-weighted ML saving calculation are unchanged. Benchmark values and benchmark-based validation verdicts can change because the confidential comparison method has been replaced rather than renamed.
