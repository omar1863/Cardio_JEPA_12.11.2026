"""Regenerate the embedded-source notebook after updating the package or driver."""
from pathlib import Path
import base64, io, json, zipfile
root=Path(__file__).resolve().parents[1]
n=json.loads((root/'notebooks/cardio_jepa_kaggle.ipynb').read_text())
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted((root/'cardio_jepa').rglob('*.py')):
        z.writestr(str(path.relative_to(root)),path.read_bytes())
payload=base64.b64encode(buf.getvalue()).decode()
source='''# Embedded corrected project source (no separate Kaggle Model input needed).
import base64, io, zipfile, os
from pathlib import Path
EMBEDDED_ROOT = Path('/kaggle/working/cardio_jepa_fixed_source') if Path('/kaggle/working').exists() else Path.cwd() / 'cardio_jepa_fixed_source'
EMBEDDED_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(EMBEDDED_ROOT)
print('Using corrected source:', EMBEDDED_ROOT)
'''.replace('PAYLOAD',repr(payload))
n['cells'].insert(1,dict(cell_type='code',id='embedded-source',metadata={},execution_count=None,outputs=[],source=source.splitlines(True)))
for cell in n['cells']:
    s=''.join(cell['source'])
    if 'PKG_ROOT = find_package_root' in s:
        s=s.replace('PKG_ROOT = find_package_root([str(f.parent) for f in _fixed] + PACKAGE_SEARCH_ROOTS)', 'PKG_ROOT = str(EMBEDDED_ROOT)')
    cell['source']=s.splitlines(True)
out=root/'notebooks/Cardio_JEPA_Kaggle_Standalone.ipynb'
out.write_text(json.dumps(n,indent=1))
print(out)
