# MIPD methodology: ML estimation of package-level post-DD FOC saving

## 1. Scope and research question

This prototype estimates the package-level change in main-engine fuel-oil consumption (FOC) associated with a dry-dock event. It does not allocate the result among individual maintenance or retrofit work items.

The research question is:

> For comparable post-DD noon reports, how much lower or higher was reported main-engine FOC than the FOC expected from a Huber model trained only on the vessel's pre-DD operating data?

The output is an observational screening estimate. It is not proof that dry docking alone caused the complete difference.

The built-in synthetic demonstration uses generated noon-report-like values with a declared approximately 10% post-DD reduction. It is provided only to explain and test the POC workflow. It is not vessel evidence and must not support a real saving claim.

The app's automatic graph interpretations are descriptive outputs calculated from the current assessment. They report operating-support coverage and the distribution of report-level actual-versus-expected differences. They do not alter the ML estimate, replace the numerical support test or establish that dry docking caused the observed difference.

## 2. Raw calculations

Speed through water is calculated from the propelling interval:

\[
STW_i=\frac{D_{LOG,i}}{H_{p,i}}
\]

Fuel masses are energy-normalised to a 40.5 MJ/kg reference and converted to a 24-hour propelling rate:

\[
FOC_i=\left(\frac{\sum_f m_{if}LCV_f}{40.5}\right)\frac{24}{H_{p,i}}
\]

Each LCV should be supported by a bunker delivery note, laboratory result or documented technical source.

M/E fuel, LOG distance and propelling hours must cover the same report interval. The app checks duplicate timestamps and whether propelling hours exceed the elapsed report interval by more than a two-hour tolerance. The first report cannot be checked against a previous timestamp and remains explicitly marked as not verifiable from the workbook alone.

## 3. Eligibility rules

The same hard quality rules apply before and after dry docking:

- propelling hours at least 18 hours;
- Beaufort no greater than 4;
- positive displacement and FOC;
- valid date and interval fuel calculation; and
- physically plausible steady-sea STW from 5 to 35 kn.

The 5-35 kn rule is an error screen, not the vessel's supported operating range. V9.11 does not use the former universal 13-25 kn operating filter. Vessel-specific operating support is learned from valid pre-DD speed and displacement observations after these hard checks.

The preferred pre-DD history is 12 months. The primary post-DD assessment period is the first 3 months after dock-out. It may be extended to the first confirmed complete normal service cycle, up to 6 months, when the primary period does not contain that cycle. The selected endpoint must be declared before the saving result is viewed.

## 4. Candidate Huber ML specifications

V9.11 compares two Huber specifications using pre-DD data only.

Speed-only candidate:

\[
\ln(FOC_i)=\beta_0+\beta_v\ln(STW_i)+\epsilon_i
\]

Speed-and-displacement candidate:

\[
\ln(FOC_i)=\beta_0+\beta_v\ln(STW_i)+\beta_\Delta\ln(\Delta_i/\Delta_{ref})+\epsilon_i
\]

The implementation uses `HuberRegressor(epsilon=1.35, alpha=0.0001)`. Huber loss limits the influence of large residuals without deleting those observations.

A training-only Duan smearing factor returns log-scale predictions to arithmetic MT/day:

\[
\widehat{FOC}_i=\left(\frac{1}{n}\sum_{j=1}^{n}e^{\widehat{\epsilon}_j}\right)e^{\widehat{\ln(FOC_i)}}
\]

Post-DD reports do not train either model or influence the specification decision.

## 5. Pre-DD ML specification selection

Both candidates are tested using expanding chronological folds. Earlier pre-DD reports train a fold and strictly later pre-DD reports test it.

The speed-and-displacement candidate is selected only when:

1. the fitted displacement coefficient is non-negative in the full pre-DD fit;
2. the displacement coefficient is non-negative in every valid chronological fold;
3. MAPE is no higher than speed-only Huber;
4. RMSE is no higher than speed-only Huber; and
5. absolute bias remains within the declared 8% project gate.

Otherwise speed-only Huber is selected. This rule prevents the post-DD saving result from determining the model.

If speed-only Huber is selected, displacement remains part of the joint operating-support test even though it is not a prediction term.

## 6. Vessel-specific operating support and route hierarchy

Post-DD reports are assessed only when their joint STW and displacement condition is represented by valid pre-DD reports. The app first applies a strict comparison requiring the same user-confirmed assessment route and operating leg.

A k-nearest-neighbour distance test uses standardised log STW and log displacement. The support threshold is the 95th percentile of the within-training three-neighbour distance. A group needs at least six pre-DD reports. These are declared prototype rules, not IMO, ISO or class requirements.

The strict result is selected when it contains at least 10 comparable post-DD reports and at least 40% of eligible post-DD fuel. When it does not, the app tests an expanded comparison that may use a different route but retains the same confirmed operating leg, Beaufort no greater than 4 and the same local speed/loading support test. Exactly one basis becomes the headline result. Expanded cross-route results are capped at Preliminary because route-related current, sea-state and operational differences may remain in noon-report data.

The app does not reconstruct the exact path sailed between noon reports. Route and operating-leg labels restrict operational comparison; they are not regression coefficients.

The user interface reports the operating comparison separately from the ML prediction model:

- the **ML specification** determines how expected FOC is predicted;
- the **operating comparison method** determines which post-DD reports may contribute to the estimate; and
- the **propelling-hour-weighted aggregation** converts report-level expected and reported FOC into the package result.

Beaufort and propelling hours are eligibility rules. They are not nearest-neighbour matching variables. The numerical operating-support test uses STW and displacement within the selected route/operating-leg hierarchy.

## 7. Package-level saving calculation

For every comparable post-DD report, the selected Huber model predicts the FOC expected from the pre-DD relationship. The result is aggregated using the same post-DD reports and propelling hours:

\[
Saving(\%)=100\left[1-\frac{\sum_i FOC_{reported,i}H_{p,i}/24}{\sum_i \widehat{FOC}_{preDD,i}H_{p,i}/24}\right]
\]

Individual report percentages are not averaged. A positive result means reported post-DD fuel was lower than model-expected fuel over the same comparable intervals.

## 8. Gross FOC review and sensitivity

The app screens for unusually large differences between reported FOC and the expectation learned from pre-DD data. A report is flagged only when both conditions are met:

- its log residual is more than 4.5 robust residual-scale units from the pre-DD centre; and
- reported FOC is below one quarter or above four times the model expectation.

These deliberately broad limits target probable unit, formula or source-record problems rather than normal noon-report scatter. Flagged reports remain visible and remain in the primary calculation. The app separately refits and recalculates after removing flagged reports, so the user can see whether the conclusion depends on them. The flag is a review prompt, not proof that the source report is wrong.

## 9. Public benchmark and sensitivity checks

The app uses the public cubic-speed rule of thumb as a simple physics-based benchmark:

\[
FOC=aV^3
\]

The vessel-specific coefficient `a` is fitted using pre-DD reports only. The selected Huber specification and cubic-speed benchmark are evaluated on the same unseen pre-DD reports and the same comparable post-DD reports. The benchmark does not replace the primary ML method and the two saving estimates are not added. It is a public rule of thumb rather than an ISO-prescribed FOC-saving calculation. IMO GreenVoyage2050 describes the cube-of-speed relationship as a rule of thumb for displacement ships: https://greenvoyage2050.imo.org/technology/speed-management/

The app also calculates the alternative Huber specification on the same comparable post-DD reports. Stability is reviewed across the selected model, alternative Huber specification, strict-versus-expanded route basis where available, anomaly-excluded refit, and a leave-one-post-DD-report-out range. These results form a sensitivity comparison, not a statistical confidence interval.

`Stable` means the available tested results retain the same direction and no gross FOC report is flagged. It does not guarantee that the estimated magnitude is tightly bounded. `Sensitive but direction stable` means all tested estimates remain positive or all remain negative, but gross FOC reports require engineering review. `Unstable` means at least one relevant sensitivity changes the direction of the conclusion or the tested range crosses zero.

## 10. Evidence checks

The prototype reports:

1. chronological validation MAPE, bias and RMSE;
2. selected-versus-alternative Huber specification evidence;
3. vessel-specific joint STW/displacement operating support;
4. strict versus expanded route-comparison basis;
5. comparable post-DD fuel coverage;
6. fake pre-DD intervention-date screening;
7. complete-service-cycle confirmation;
8. operating-scope concentration by route and leg;
9. source-reference completeness;
10. gross FOC anomaly review and anomaly-excluded sensitivity;
11. model, route and individual-report stability; and
12. a complete report inclusion and exclusion audit.

The supported, indicative, preliminary, unstable and inconclusive categories are prototype screening classifications. They are not certification or statistical confidence levels. `Supported prototype screening` requires the strongest declared evidence gates, including confirmed dates, route and service cycle, strict same-route comparison, acceptable unseen-data validation and a direction-consistent sensitivity result. `Indicative` requires acceptable model validation, strict same-route evidence, at least 20 comparable post-DD reports and at least 70% comparable fuel coverage. `Preliminary` is reserved for a validated calculation with limitations such as expanded cross-route comparison, incomplete operational confirmation or limited post-DD evidence. `Unstable` means reasonable alternative analyses change the direction between saving and higher FOC. `Inconclusive` applies when the model is invalid, its validation is unavailable or outside the declared MAPE, bias or speed-exponent gates, or the minimum calculation floor is not met. Expanded cross-route evidence cannot exceed `Preliminary`.

## 11. Method scope, standards reference and interpretation boundary

The result applies only to the disclosed comparable operating conditions and represents steady at-sea main-engine performance, not total voyage fuel. A sample dominated by one route or operating leg must not be presented as representative of unsupported operations. Beaufort filtering reduces heavy-weather influence but does not fully weather-correct the result. The analysis does not prove causation and cannot attribute the package-level difference to individual coatings, cleaning work, propeller work or retrofits.

The prototype follows the general engineering principle of comparing the same vessel under comparable operating conditions. It does not claim ISO 19030 conformance and does not implement the ISO 19030 default performance-indicator calculation, reference displacement correction or continuous sensor-data requirements. Its adjusted-baseline logic is consistent with the general measurement-and-verification concept described by ISO 50015, but the prototype does not claim ISO 50015 certification or conformance. Its speed/loading support test and evidence tiers are project-specific safeguards for noon-report ML analysis.

Public references:

- ISO 19030-2 overview: https://www.iso.org/standard/63775.html
- ISO 50015 overview: https://www.iso.org/standard/60043.html
- IMO GreenVoyage2050 speed-management rule of thumb: https://greenvoyage2050.imo.org/technology/speed-management/
