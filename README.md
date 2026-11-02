# MIPD V9.14 — Streamlit Community Cloud

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

## What changed in V9.14

- Separates assessed FOC saving direction from reliability: positive and negative estimates can be **Supported Assessment**.
- Requires ≥60 eligible pre-DD and ≥20 comparable post-DD reports; ≥60% matched post-DD fuel; MAPE ≤12%; |bias| ≤8%; stable sensitivity direction; confirmed route, service-cycle, period and reporting interval.
- At least 10 comparable post-DD reports and 40% matched fuel remain the minimum for an exploratory result.
- Two-sided placebo-effect magnitude is displayed as a separate effect-distinguishability diagnostic, not a data-quality grading gate.
- The calculation continues using the vessel-specific pre-DD Huber model and pre-DD operating support; assessment labels are not proof of a causal dry-docking effect.

## Project files

This branch restores the original MIPD noon-report processing and HTML report modules that were accidentally replaced by files from an unrelated Operating Profile app in the default branch. The main default branch is left unchanged.

## Verification

15 V9.14 reliability unit tests passed in the preparation environment. Full Streamlit Community Cloud runtime and integration testing are pending deployment. The eight-vessel comparison is provisional pending verification of operating mappings and service-cycle evidence.

## Confidentiality and operational notes

Do not assume uploads are approved merely because they can be processed. Public app visitors may open the same deployed app. Community Cloud is not a substitute for your organisation's data-handling rules.
