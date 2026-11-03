"""A minimal external player using only the SDK. No engine installation needed."""

import argparse
import getpass

from battleship_client import BattleshipClient
from battleship_client.policies import play_policy
from battleship_protocol import Coordinate, Observation


class FirstLegalPlayer:
    def choose_shot(self, observation: Observation) -> Coordinate:
        for row, cells in enumerate(observation.legal_shot_mask):
            for column, legal in enumerate(cells):
                if legal:
                    return Coordinate(row=row, column=column)
        raise ValueError("No legal shot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("match_id")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with BattleshipClient(args.url) as client:
        client.join(args.match_id, getpass.getpass("Invitation: "))
        final = play_policy(client, FirstLegalPlayer())
        print(final.result.model_dump_json())


if __name__ == "__main__":
    main()
