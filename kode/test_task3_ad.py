import ast
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

nb=json.loads(Path(__file__).with_name('Task3_Clustering.ipynb').read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell['cell_type']=='code':
        tree=ast.parse(''.join(cell['source']))
        defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['numeric_block','composition_block','weight_block']]
        if defs: exec(compile(ast.Module(body=defs,type_ignores=[]),'<notebook-functions>','exec'),globals())

class PreprocessChecks(unittest.TestCase):
    def test_composition_zeros_and_closure(self):
        x=pd.DataFrame([[0.,0.],[2.,2.],[0.,4.]])
        z=composition_block(x)
        np.testing.assert_allclose((z*z).sum(axis=1),1)
        np.testing.assert_allclose(z[1],[np.sqrt(.5),np.sqrt(.5)])
        self.assertEqual(z[2,0],0)
    def test_negative_share_rejected(self):
        with self.assertRaises(ValueError):composition_block(pd.DataFrame([[-1.,2.]]))
    def test_numeric_does_not_modify_raw(self):
        x=pd.DataFrame({'value':[0.,1.,np.nan,1e10],'constant':[3.]*4})
        saved=x.copy(deep=True)
        z,rows,_=numeric_block(x)
        self.assertTrue(np.isfinite(z).all())
        pd.testing.assert_frame_equal(x,saved)
        self.assertEqual(rows[0]['missing'],1)
    def test_block_weight_and_constant(self):
        z=weight_block(np.array([[0.,1.],[2.,3.]]),.25)
        self.assertAlmostEqual(np.mean(np.sum(z*z,axis=1)),.25)
        self.assertTrue(np.isfinite(weight_block(np.ones((3,2)))).all())

if __name__=='__main__':unittest.main()
