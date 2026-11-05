import sys
import types
import itertools

# Mock pytest so we don't depend on pytest binary
mock_pytest = types.ModuleType('pytest')
class Mark:
    @staticmethod
    def parametrize(*args, **kwargs):
        def dec(f): return f
        return dec
mock_pytest.mark = Mark()
sys.modules['pytest'] = mock_pytest

sys.path.insert(0, '.')
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')

from test_multichannel_fine_tune import test_multichannel_fine_tune

fine_tuning_types = ['head', 'full']
adapters = [None, 'pca', 'lcomb']
device = 'cpu'

print("=" * 70)
print("Running test_multichannel_fine_tune across all 6 parameter configurations")
print("=" * 70)

results = []
for ft_type, adapter in itertools.product(fine_tuning_types, adapters):
    adapter_str = str(adapter)
    print(f"\n--- Testing fine_tuning_type='{ft_type}', adapter={adapter_str}, device='{device}' ---")
    try:
        test_multichannel_fine_tune(ft_type, adapter, device)
        results.append((ft_type, adapter_str, "PASSED"))
    except Exception as e:
        print(f"FAILED: {e}")
        results.append((ft_type, adapter_str, f"FAILED: {e}"))

print("\n" + "=" * 70)
print("TEST SUMMARY:")
print("=" * 70)
for ft_type, adapter_str, status in results:
    print(f"  • fine_tuning_type={ft_type:6s} | adapter={adapter_str:6s} ==> {status}")
print("=" * 70)
