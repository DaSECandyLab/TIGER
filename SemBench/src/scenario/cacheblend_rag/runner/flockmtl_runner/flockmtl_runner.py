from pathlib import Path
import sys

from runner.generic_flockmtl_runner.generic_flockmtl_runner import GenericFlockMTLRunner
from scenario.cacheblend_rag.setup.flockmtl import FlockMTLCacheBlendRagSetup

sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class FlockMTLRunner(GenericFlockMTLRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup=skip_setup,
        )
        setup = FlockMTLCacheBlendRagSetup(model_name=model_name)
        setup.setup_data()
        self.flockmtl_conn = setup.get_connection()
