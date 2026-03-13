# Data files

The current script uses root-level relative file paths. Unless you refactor the code, place the following files in the repository root before running:

- `NIST_WaterCrossSections.csv`
- `OxygenPhotoShells.csv`
- `ElectronStoppingPower.csv`
- `Final_cross_sections.csv`
- `Rayleigh_cross_sections.csv`

Optional runtime/generated files:

- `mc_physics_data.npz`
- `physics_head_pretrained.pth`
- `replay_buffer.pkl`
- `hybrid_sac_model.zip`

If you prefer a cleaner layout, move the files into `data/` and update the paths in `Rad_sac_24_maio_fixed.py`.
