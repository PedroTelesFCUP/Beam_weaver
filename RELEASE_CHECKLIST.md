# Release checklist

## Recommended publication route

### Route A — GitHub + Zenodo integration
Use this when you want your GitHub release archived automatically by Zenodo.

1. Create a public GitHub repository.
2. Upload the files in this release kit.
3. Add your repository license on GitHub.
4. Connect your GitHub account to Zenodo.
5. Enable the repository in Zenodo.
6. Push a tagged release on GitHub, e.g. `v0.1.0`.
7. Wait for Zenodo to archive the release and mint the DOI.

**Important:** in the GitHub-integrated flow, Zenodo does **not** let you pre-reserve the DOI before the GitHub release is created.

### Route B — Manual Zenodo upload
Use this when you need the DOI **before** publication so you can place it inside files.

1. Create a new Zenodo upload manually.
2. Click **Get a DOI now!** to reserve the DOI.
3. Insert that DOI into:
   - `README.md`
   - `CITATION.cff`
   - release notes
4. Upload the repository snapshot as a ZIP file.
5. Publish the record.

## Files to review before first public release

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `.zenodo.json`
- `requirements.txt`
- `environment.yml`
- `.gitignore`

## Fields to personalize

Before publishing, update:

- author name(s)
- ORCID(s)
- affiliation(s)
- GitHub repository URL
- project version
- abstract text if desired
- any funding information
- any paper DOI / preprint DOI
- Zenodo community (optional)

## After DOI minting

Update the repository with:

- DOI badge in `README.md`
- DOI field in `CITATION.cff`
- `related_identifiers` in `.zenodo.json` if you want to link the paper and software
- GitHub “About” description and website link

## Suggested first tag

- `v0.1.0` for a proof-of-concept public release

## Strongly recommended before wider dissemination

- remove any private or bulky files from the repository
- verify that you are allowed to redistribute each data table
- add one tiny example dataset or a smoke test
- confirm the environment versions used to produce the published figures
- write one short `CHANGELOG.md` for later releases
