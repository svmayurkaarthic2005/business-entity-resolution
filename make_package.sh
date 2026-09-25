#!/bin/bash
# Build <team>_submission.zip in the layout the organisers require:
#   output/{matching_results,candidate_pairs}.tsv
#   code/business_entity_resolution/{src/,README.md,requirements.txt,config.yaml,run_all.sh}
#   Documentation_template.md
set -e
cd "$(dirname "$0")"
TEAM=${TEAM:-team}
PKG=dist/${TEAM}_submission
rm -rf "$PKG" && mkdir -p "$PKG/output" "$PKG/code/business_entity_resolution"
cp output/matching_results.tsv output/candidate_pairs.tsv "$PKG/output/"
cp -r src README.md requirements.txt config.yaml run_all.sh "$PKG/code/business_entity_resolution/"
rm -rf "$PKG/code/business_entity_resolution/src/__pycache__"
cp Documentation_template.md "$PKG/"
(cd dist && rm -f "${TEAM}_submission.zip" && zip -qr "${TEAM}_submission.zip" "${TEAM}_submission")
ls -la "dist/${TEAM}_submission.zip"
