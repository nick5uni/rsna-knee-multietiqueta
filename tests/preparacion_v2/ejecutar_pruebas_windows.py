"""Ejecuta los controles de preparación con la escritura LF de Linux/Kaggle."""
from pathlib import Path
from unittest.mock import patch
import runpy
original=Path.write_text
def linux_write(self,data,*args,**kwargs):
    kwargs.setdefault('newline','\n')
    return original(self,data,*args,**kwargs)
with patch.object(Path,'write_text',linux_write):
    runpy.run_path(str(Path(__file__).with_name('test_preparacion_celda_unica.py')),run_name='__main__')
