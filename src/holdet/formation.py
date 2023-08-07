from rich.table import Table

from holdet.candidate import Candidate


class Formation:
    def __init__(self, solution: list[Candidate]):
        self.position: dict[str, list[Candidate]] = {
            "keeper": [],
            "defenses": [],
            "midfielders": [],
            "forwards": [],
        }
        self._populate(solution)

    def _populate(self, solution: list[Candidate]):
        for player in solution:
            if player.keeper:
                self.position["keeper"].append(player)
            if player.defense:
                self.position["defenses"].append(player)
            if player.midfielder:
                self.position["midfielders"].append(player)
            if player.forward:
                self.position["forwards"].append(player)

    def __iter__(self):
        return iter(self.position.items())

    def __repr__(self) -> str:
        return (
            f"{len(self.position['defenses'])}-"
            f"{len(self.position['midfielders'])}-"
            f"{len(self.position['forwards'])}"
        )

    def __rich__(self) -> Table:
        table = Table(title=f"\nXI ({self})")
        table.add_column("Position")
        table.add_column("Players")
        table.add_column("xGrowth", justify="right")

        # Keep track of value and xGrowth totals
        total_xGrowth = 0
        total_value = 0

        for index, (position, players) in enumerate(self):
            for player in players:
                # Calculate the xGrowth by subtracting the player's expected
                # value from the actual value.
                xGrowth = player.xValue - player.value

                total_xGrowth += xGrowth
                total_value += player.value

                table.add_row(
                    position.capitalize(),
                    str(player),
                    f"{xGrowth / 1000:.0f}K",
                    # End the section after the last player is added
                    end_section=True if index == 10 else False,
                )

                # Avoid repeating the position name in the same row
                position = ""

            # Add an empty row between positions
            if index < 3:
                table.add_row()

        # Add a total row
        table.add_row(
            "Total",
            f"value={total_value / 1000000:.1f}M",
            f"{total_xGrowth / 1000:.0f}K",
        )

        return table
