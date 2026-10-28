import sys
import tarfile
import io
from types import SimpleNamespace

from diagnostics.standard_preflight import download_imagenet_classes


def test_cbp_download_extracts_required_classes_and_reuses_cache(tmp_path,monkeypatch):
    calls=[]
    def download(**kwargs):
        calls.append(kwargs['id'])
        with tarfile.open(kwargs['output'],'w:gz') as archive:
            for name,data in [('data/2.npy',b'class2'),('data/3.npy',b'class3'),('data/99.npy',b'excluded')]:
                info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        return kwargs['output']
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=download))
    from pathlib import Path
    target=Path(download_imagenet_classes(tmp_path,{'2.npy','3.npy'}))
    assert {p.name for p in target.iterdir()}=={'2.npy','3.npy'}
    assert (target/'2.npy').read_bytes()==b'class2'
    assert not (tmp_path/'cbp_imagenet32.download').exists()
    assert download_imagenet_classes(tmp_path,{'2.npy','3.npy'})==str(target)
    assert len(calls)==1
