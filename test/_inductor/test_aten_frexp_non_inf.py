# Union frexp test for the IEEE-754 field-extraction fix (torch_npu inductor lowering).
#
# torch_npu/_inductor/runtime/triton_helpers.py now computes frexp by IEEE-754
# field extraction instead of libdevice ilogb/ldexp (which were type-mismatched
# and mis-rounded at 2^k). This test is the union of the previous frexp tests plus
# The review edge cases:
# - all frexp op overloads: aten.frexp/.Tensor/.Tensor_out, prims.frexp/.default
# - no regression on normal float32/float16/bfloat16 (asserts mantissa AND exponent;
# the original test only checked exponent)
# - exact 2^k boundary values, both signs (the old ilogb rounding defect)
# -inf / -inf / nan / +-0
# - small fixed tensor and a fused frexp+add path
import torch
import torch_npu
from torch.testing._internal.common_utils import (
    run_tests, parametrize, instantiate_parametrized_tests, TestCase,
)

FLOATS = [ 'float32' , 'float16' , 'bfloat16' ]


def  _aten ( x ):
    return torch.ops.aten.frexp(x)

def  _aten_tensor ( x ):
    return torch.ops.aten.frexp.Tensor(x)

def  _aten_tensor_out ( x ):
    m = torch.empty_like(x)
    e = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    torch.ops.aten.frexp.Tensor_out(x, mantissa=m, exponent=e)
    return m, e

def  _prims ( x ):
    return torch.ops.prims.frexp(x)

def  _prims_default ( x ):
    return torch.ops.prims.frexp.default(x)

OPS = {
    'aten' : _aten,
    'aten_tensor' : _aten_tensor,
    'aten_tensor_out' : _aten_tensor_out,
    'prims' : _prims,
    'prims_default' : _prims_default,
}


class  TestFrexpNonInf ( TestCase ):

    def  _compare ( self, fn, x, equal_nan= False ):
        std_m, std_e = fn(x)
        ind_m, ind_e = torch. compile (fn, backend= "inductor" )(x)
        print ( f"\nx\n {x} \nstd_mantissa\n {std_m} \ninductor_mantissa\n {ind_m} "
              f"\nstd_exponent\n {std_e} \ninductor_exponent\n {ind_e} " )
        self.assertEqual(std_m, ind_m, equal_nan=equal_nan)
        self.assertEqual(std_e, ind_e)

    # every frexp overload, on a normal random tensor, per float dtype
    @parametrize( 'op' , list ( OPS.keys() ) )
    @parametrize( 'dtype' , FLOATS )
    def  test_frexp_op_variants ( self, op, dtype ):
        torch.manual_seed( 0 )
        x = torch.randn(( 256 , 256 ), dtype= eval ( 'torch.' + dtype), device= 'npu' ) * 2000
        self._compare(OPS[op], x)

    # no regression on a large random input (the mantissa bug we fixed)
    @parametrize( 'dtype' , FLOATS )
    def  test_frexp_normal_random ( self, dtype ):
        torch.manual_seed( 0 )
        x = torch.randn(( 128 , 1024 , 4096 ), dtype= eval ( 'torch.' + dtype), device= 'npu' ) * 2000
        self._compare(_aten, x)

    # exact 2^k boundaries, both signs (the old ilogb rounding defect)
    def  test_frexp_pow2_boundaries ( self ):
        vals = []
        for k in  range ( -40 , 40 ):
            p = 2.0 ** k
            vals += [p, -p]
        x = torch.tensor(vals, dtype=torch.float32, device= 'npu' )
        self._compare(_aten, x)

    # inf / -inf / nan / +-0
    def  test_frexp_inf_nan_zero ( self ):
        x = torch.tensor([ float ( 'inf' ), float ( '-inf' ), float ( 'nan' ),
                          0.0 , -0.0 , 1.0 , -3.5 ],
                         dtype=torch.float32, device= 'npu' )
        self._compare(_aten, x, equal_nan= True )

    # small fixed tensor, aten + prims, per dtype
    @parametrize( 'dtype' , FLOATS )
    def  test_frexp_small_fixed ( self, dtype ):
        x = torch.tensor([[ 1.0 , 2.0 , 3.0 , 4.0 ], [ 5.0 , 6.0 , 7.0 , 8.0 ]],
                         dtype= eval ( 'torch.' + dtype), device= 'npu' )
        self._compare(_aten, x)
        self._compare(_prims_default, x)

    # fused frexp + add path (mantissa/exponent both offset by 1)
    @parametrize( 'dtype' , FLOATS )
    def  test_frexp_fused_add ( self, dtype ):
        def  fused ( t ):
            m, e = torch.ops.aten.frexp(t)
            return m + 1 , e + 1
        torch.manual_seed( 0 )
        x = torch.randn(( 512 , 512 ), dtype= eval ( 'torch.' + dtype), device= 'npu' ) * 2000
        self._compare(fused, x)


instantiate_parametrized_tests(TestFrexpNonInf)

if __name__ == "__main__" :
    run_tests()