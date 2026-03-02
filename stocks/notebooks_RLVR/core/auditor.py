import pandas as pd
from core.quant import QuantUtils
from core.result import TaskResult


class SystemAuditor:
    @staticmethod
    def verify_math_integrity() -> TaskResult:
        """
        🛡️ TRIPWIRE: Validates math kernels before execution.
        """
        try:
            # Test 1: Series Boundary
            mock_s = pd.Series([100.0, 102.0, 101.0])
            rets_s = QuantUtils.compute_returns(mock_s)
            if not pd.isna(rets_s.iloc[0]):
                return TaskResult(
                    ok=False, msg="Math Integrity: Series Leading NaN missing"
                )

            # Test 2: DataFrame Boundary
            mock_df = pd.DataFrame({"A": [100, 101], "B": [200, 202]})
            rets_df = QuantUtils.compute_returns(mock_df)
            if not rets_df.iloc[0].isna().all():
                return TaskResult(
                    ok=False, msg="Math Integrity: DF Leading NaN missing"
                )

            return TaskResult(ok=True, msg="Mathematical boundaries strictly enforced.")

        except Exception as e:
            return TaskResult(ok=False, msg=f"System Breach during audit: {str(e)}")


#
