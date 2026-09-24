```python
Data
Place the two study datasets in this folder so that every command in
the repository works out of the box with the default relative paths.

Required files (commit these to GitHub)
text

data/
├── SDBNew Latlong.xlsx      ← field observations (CTD/survey points)
└── 20220213_SDB_5bands.tif  ← Sentinel-2 5-band scene used for prediction
After copying them here the repository layout is:

text

SDB-Framework/
├── sdb_framework_final.py
├── data/
│   ├── README.md
│   ├── SDBNew Latlong.xlsx
│   └── 20220213_SDB_5bands.tif
└── ...
and the default configuration inside SDBConfig already points at them:

Python

excel_path: Path = Path("data/SDBNew Latlong.xlsx")
raster_path: Path = Path("data/20220213_SDB_5bands.tif")
output_dir: Path = Path("outputs/full_run")
so a reviewer needs no arguments at all:

Bash

python sdb_framework_final.py --mode full
File 1 — SDBNew Latlong.xlsx
Column	Type	Unit / CRS	Notes
En_Name	string	–	Unique station identifier
Longitude	float	degrees, EPSG:4326	Decimal degrees
Latitude	float	degrees, EPSG:4326	Decimal degrees
Depth	float	metres	Positive depth; framework design range 0–3 m
Common aliases (Lon/Lat, Long/Latitude, water_depth, …) are
auto-detected by RasterSampler. A CSV with the same columns is also
accepted — just pass --excel-path explicitly.

File 2 — 20220213_SDB_5bands.tif
Format: GeoTIFF
Bands: exactly 5, in order B2, B3, B4, B8, B11 (Sentinel-2 MSI)
Reflectance: integer scaling auto-detected (divisor 10000) or override
with reflectance_scale in the configuration
CRS: any CRS supported by rasterio; field points (EPSG:4326) are
reprojected automatically during sampling
The raster must spatially overlap the field points
Size check before uploading to GitHub
GitHub rejects individual files larger than 100 MB:

Bash

ls -lh data/
Both files < 100 MB → commit directly (normal case for a clipped
scene and a spreadsheet).
GeoTIFF ≥ 100 MB → use Git LFS:
Bash

git lfs install
git lfs track "data/*.tif"
git add .gitattributes
…or upload the raster to Zenodo (free DOI),
commit only data/README.md, and fill in the download link below.
```
