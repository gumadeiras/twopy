# example_recording_small

Small real-data fixture derived from Gustavo's example converted recording:

`/Users/gumadeiras/Yale University Dropbox/users/gustavo_santana/data/2p_microscope_data/gh146gal4_w;uasg6f_+;orcolexa,lexaopcschrimsontdtomato_+/combo_stim_singles=3s_blank=3s_intensity=20/2023/10_17/10_02_49/twopy`

Contents:

- `recording_data.h5`: converted metadata, stimulus sample rows, photodiode
  sample rows, mean image, and frame-count audit metadata.
- `aligned_movie.h5`: frames `54:114` and spatial crop
  `axis0=[6, 30)`, `axis1=[9, 33)` from the converted aligned movie.
- `rois.h5`: three small ROI masks drawn on the cropped fixture frame.

The fixture is intentionally tiny. It is not a scientific analysis output; it
is a regression fixture that keeps loader, ROI, workflow, and napari adapter
tests anchored to real pixel values without committing a full recording.
