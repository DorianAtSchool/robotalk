import unittest

from robocasa.utils.sim_tool_specs import (
    SIM_TOOL_SPEC_BY_NAME,
    SIM_TOOL_SPECS,
    get_sim_tool_spec,
    get_sim_tool_specs,
)


class TestSimToolSpecs(unittest.TestCase):

    def test_no_duplicate_tool_names(self):
        names = [spec["name"] for spec in SIM_TOOL_SPECS]
        self.assertEqual(len(names), len(set(names)))


    def test_getters_return_defensive_copies(self):
        all_specs = get_sim_tool_specs()
        all_specs[0]["name"] = "mutated"
        self.assertNotEqual(SIM_TOOL_SPECS[0]["name"], "mutated")

        single_spec = get_sim_tool_spec("navigate_to_fixture")
        single_spec["parameters"][0]["name"] = "mutated_fixture_id"
        self.assertEqual(
            SIM_TOOL_SPEC_BY_NAME["navigate_to_fixture"]["parameters"][0]["name"],
            "fixture_id",
        )

    def test_executor_has_method_for_every_tool(self):
        """Every tool in the spec registry must have a matching method on SimToolExecutor."""
        from robocasa.utils.sim_tool_executor import SimToolExecutor

        for tool_name in SIM_TOOL_SPEC_BY_NAME:
            self.assertTrue(
                hasattr(SimToolExecutor, tool_name),
                f"SimToolExecutor missing method for {tool_name!r}",
            )
            self.assertTrue(callable(getattr(SimToolExecutor, tool_name)))


if __name__ == "__main__":
    unittest.main()
