import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('independent_verify',Path(__file__).with_name('verify.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class IndependentGeometry(unittest.TestCase):
    def test_actual_points_not_boxes_and_local_slope_height(self):
        triangles=np.array([[[0,0,0],[10,0,10],[0,10,0]],[[10,0,10],[10,10,10],[0,10,0]]],dtype=float)
        points=np.array([[1,1,1],[1,1,9],[11,1,11],[5,5,5.5],[5,5,6.1]])
        self.assertEqual(module.core_support(points,triangles,1).tolist(),[True,False,False,True,False])
    def test_binary_ply_keeps_original_coordinates_and_fields(self):
        header=b'ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty double x\nproperty double y\nproperty double z\nproperty float change_object\nend_header\n'
        dtype=np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('change_object','<f4')]);points=np.array([(1,2,3,1),(4,5,6,np.nan)],dtype=dtype)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.ply';path.write_bytes(header+points.tobytes())
            self.assertEqual(module.ply_vertices(path)['x'].tolist(),[1,4])
            self.assertTrue(np.isnan(module.ply_vertices(path)['change_object'][1]))
            path.write_bytes(header+points.tobytes()[:-1])
            with self.assertRaises(ValueError):module.ply_vertices(path)
if __name__=='__main__':unittest.main()
