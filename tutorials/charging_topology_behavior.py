"""Demonstrate how EV2Gym assigns chargers to transformers/grid nodes.

Run from the repository root with:
    python -m tutorials.charging_topology_behavior
"""

import json
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

from ev2gym.models.ev2gym_env import EV2Gym


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = ROOT / "ev2gym/example_config_files/charging_topology_two_areas.json"


def make_config(source: Path, destination: Path, **overrides) -> dict:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config.update(overrides)
    destination.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config


def main() -> None:
    with TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)

        # The YAML declares 2 charging stations, while the JSON contains 3.
        topology_config_path = temporary_directory / "topology.yaml"
        topology_config = make_config(
            ROOT / "ev2gym/example_config_files/simplePST.yaml",
            topology_config_path,
            charging_network_topology=str(TOPOLOGY),
            number_of_charging_stations=2,
            number_of_transformers=1,
            simulation_length=24,
            random_day=False,
            simulate_grid=False,
        )
        topology_env = EV2Gym(config_file=topology_config_path, seed=7)

        print("Topology-only mode")
        print("  YAML charging stations:", topology_config["number_of_charging_stations"])
        print("  Effective charging stations:", topology_env.cs)
        print("  Effective transformers:", topology_env.number_of_transformers)
        print(
            "  Charger -> transformer:",
            [(cs.id, cs.connected_transformer) for cs in topology_env.charging_stations],
        )

        assert topology_env.cs == 3
        assert topology_env.number_of_transformers == 2
        assert [cs.connected_transformer for cs in topology_env.charging_stations] == [
            0,
            0,
            1,
        ]
        topology_env.step(np.zeros(topology_env.number_of_ports))
        assert topology_env.current_step == 1

        # Without a topology, grid-mode chargers remain round-robin distributed.
        grid_config_path = temporary_directory / "grid.yaml"
        grid_config = make_config(
            ROOT / "ev2gym/example_config_files/V2Ggrid.yaml",
            grid_config_path,
            charging_network_topology="None",
            number_of_charging_stations=36,
            simulation_length=24,
            random_day=False,
            simulate_grid=True,
        )
        grid_env = EV2Gym(config_file=grid_config_path, seed=7)
        bus_data = pd.read_csv(grid_config["network_info"]["bus_info_file"])
        non_slack_bus_labels = bus_data.loc[bus_data["Tb"] != 1, "NODES"].tolist()
        charger_bus_labels = [
            non_slack_bus_labels[cs.connected_bus] for cs in grid_env.charging_stations
        ]

        print("Grid mode")
        print("  Grid nodes (including slack):", grid_env.grid.node_num)
        print("  Effective transformers/non-slack nodes:", grid_env.number_of_transformers)
        print("  First 36 charger bus labels:", charger_bus_labels)

        assert grid_env.number_of_transformers == 33
        assert charger_bus_labels == list(range(2, 35)) + [2, 3, 4]
        grid_env.step(np.zeros(grid_env.number_of_ports))
        assert grid_env.current_step == 1

        print("  One simulation step completed in each valid mode")

        # The topology explicitly selects buses 5 and 17.
        combined_config_path = temporary_directory / "topology_with_grid.yaml"
        make_config(
            ROOT / "ev2gym/example_config_files/V2Ggrid.yaml",
            combined_config_path,
            charging_network_topology=str(TOPOLOGY),
            number_of_charging_stations=2,
            simulation_length=24,
            random_day=False,
            simulate_grid=True,
        )
        print("Topology JSON + grid mode")
        for seed in (1, 7, 23, 101):
            combined_env = EV2Gym(config_file=combined_config_path, seed=seed)
            combined_bus_labels = [
                non_slack_bus_labels[cs.connected_bus]
                for cs in combined_env.charging_stations
            ]
            while not combined_env.done:
                combined_env.step(np.ones(combined_env.number_of_ports))
            active_grid_rows = np.flatnonzero(
                np.any(combined_env.node_ev_power != 0, axis=1)
            ).tolist()

            print(f"  Seed {seed}: buses {combined_bus_labels}, active rows {active_grid_rows}")
            assert combined_env.cs == 3
            assert combined_env.number_of_transformers == 2
            assert combined_env.cs_transformers == [0, 0, 1]
            assert combined_bus_labels == [5, 5, 17]
            assert active_grid_rows and set(active_grid_rows) <= {4, 16}
            assert combined_env.stats is not None

        invalid_topology = json.loads(TOPOLOGY.read_text(encoding="utf-8"))
        invalid_topology["area_south"]["bus_id"] = 999
        invalid_topology_path = temporary_directory / "invalid_topology.json"
        invalid_topology_path.write_text(json.dumps(invalid_topology), encoding="utf-8")
        invalid_config_path = temporary_directory / "invalid_bus.yaml"
        make_config(
            ROOT / "ev2gym/example_config_files/V2Ggrid.yaml",
            invalid_config_path,
            charging_network_topology=str(invalid_topology_path),
            simulation_length=24,
            random_day=False,
            simulate_grid=True,
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                EV2Gym(config_file=invalid_config_path, seed=7)
            except ValueError:
                pass
            else:
                raise AssertionError("An unavailable bus ID must stop initialization")
        assert any("999" in str(warning.message) for warning in caught_warnings)
        print("  Invalid bus 999: warning emitted and initialization stopped")


if __name__ == "__main__":
    main()
