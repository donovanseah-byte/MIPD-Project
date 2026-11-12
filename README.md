# MIPD V9.15 interface — Streamlit Community Cloud

Browser-ready branch of the Maritime Innovation Project Day (MIPD) package-level post-dry-docking fuel-oil-consumption assessment.

## Run in your browser (no Command Prompt)

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Select **Create app** / **Deploy an app** → **From GitHub**.
3. Select repository: `donovanseah-byte/MIPD-Project`.
4. Select branch: `mipd-v9-14-streamlit`.
5. Set the entry-point file to `app.py`.
6. Select a public or private app as permitted by your company's data policies and deploy.
7. Use the resulting Streamlit Cloud URL to open the app in a browser.

**Data confidentiality:** Do not commit vessel noon-report workbooks, credentials or company records to GitHub. This repository is public. A hosted Streamlit instance still processes uploaded vessel data on a third-party server; get ONE's express approval before using sensitive company data. For confidential work without cloud approval, run the existing local/offline version instead.

## Current reliability framework (15-report Supported threshold)

- Separates assessed FOC saving direction from reliability: positive and negative estimates can be **Supported Assessment**.
- Requires ≥60 eligible pre-DD and ≥15 comparable post-DD reports; ≥60% matched post-DD fuel; MAPE ≤12%; |bias| ≤8%; stable sensitivity direction; confirmed route, monitoring-period coverage and reporting-interval alignment. Complete normal service-cycle verification is reported separately as Confirmed or Pending and is **not** a data/model reliability gate.
- At least 10 comparable post-DD reports and 40% matched fuel remain the minimum for an exploratory result.
- Two-sided placebo-effect magnitude is displayed as a separate effect-distinguishability diagnostic, not a data-quality grading gate.
- The calculation continues using the vessel-specific pre-DD Huber model and pre-DD operating support; assessment labels are not proof of a causal dry-docking effect.
- **No target count is hard-coded:** any vessel that passes the declared numerical/operating-comparability checks is Supported for **data/model screening**, whether its FOC estimate is positive or negative. An unconfirmed service cycle must remain labelled **Pending** and limits operational interpretation.

## Project files

This branch restores the original MIPD noon-report processing and HTML report modules that were accidentally replaced by files from an unrelated Operating Profile app in the default branch. The same analytical grading revision is now present on the default branch and this Streamlit branch.

## Verification

The Supported Assessment threshold was reduced from 20 to 15 comparable post-DD reports. The 10-report calculation floor, model fitting, FOC formula, operating-support selection, and all other reliability gates remain unchanged. The repository includes updated boundary tests; run `python -m unittest test_reliability.py -v`. Confirm the new classifications by rerunning the original eight vessel workbooks in Streamlit with unchanged settings. A code update is not proof of successful Cloud redeployment or live assessment. Missing service-cycle proof remains Pending and must not be represented as independently verified.

## Confidentiality and operational notes

Do not assume uploads are approved merely because they can be processed. Public app visitors may open the same deployed app. Community Cloud is not a substitute for your organisation's data-handling rules.
