from pathlib import Path
import urllib.request

URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269705/suppl/GSE269705_RAW.tar"

def main(out_dir: str = "data/raw"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "GSE269705_RAW.tar"
    if out_path.exists() and out_path.stat().st_size > 1_000_000_000:
        print(f"OK: {out_path} už existuje ({out_path.stat().st_size/1e9:.2f} GB)")
        return
    print("Stahuju:", URL)
    urllib.request.urlretrieve(URL, out_path)
    print("Hotovo:", out_path)

if __name__ == "__main__":
    main()
