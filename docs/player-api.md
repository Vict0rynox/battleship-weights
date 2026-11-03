# Player API and SDK — protocol 1.0

Players need only `battleship-protocol`, `battleship-client`, and this document.
Other languages can use HTTP/JSON directly. The server's `/openapi.json` contains
the complete request/response schemas; `/docs` provides interactive examples.

## Lifecycle and credentials

1. `POST /v1/matches` with `{"action_id":"unique-create-id"}` returns
   `{match_id, player:0, token, invitation}`.
2. Give the other player the match ID and invitation. They call
   `POST /v1/matches/{id}/join` with
   `{"action_id":"unique-join-id","invitation":"..."}` and receive
   `{match_id, player:1, token}`.
3. Every remaining endpoint requires `Authorization: Bearer TOKEN`.
4. Each player commits a fleet. The first player may place while awaiting a join.
5. When both are ready, player 0 shoots first. Turns alternate until victory,
   resignation, or a configured shooting deadline expires.

Credentials identify the seat; actions never accept a player index. Match IDs grant
no access. Invitations are one-use; bearer credentials remain valid for reading
that player's final view after the match. There is no public match listing.

Phases are `waiting` (second seat absent), `placement`, `playing`, and `finished`.
The revision increments on each successful state transition, including timeout.
Polling an unchanged match does not increment it.

## Placement and actions

Coordinates are zero-based `{row, column}` with row 0 at the top and column 0 at
the left. Terminal `A1` is `{row:0,column:0}`; `J10` is `{row:9,column:9}`.

Automatic placement:

```json
{"action_id":"place-1","mode":"automatic"}
```

Manual placement at `POST /v1/matches/{id}/placement`:

```json
{
  "action_id": "place-1",
  "mode": "manual",
  "ships": [
    {"name":"carrier","start":{"row":0,"column":0},"orientation":"horizontal"},
    {"name":"battleship","start":{"row":1,"column":0},"orientation":"horizontal"},
    {"name":"cruiser","start":{"row":2,"column":0},"orientation":"horizontal"},
    {"name":"submarine","start":{"row":3,"column":0},"orientation":"horizontal"},
    {"name":"destroyer","start":{"row":4,"column":0},"orientation":"horizontal"}
  ]
}
```

All five names must occur exactly once. Automatic mode must not include ships.
Placement is atomic and immutable once accepted. Touching, including diagonally,
is permitted. Direction is always positive: right for horizontal, down for vertical.

Fire at `POST /v1/matches/{id}/shots`:

```json
{"action_id":"shot-1","row":4,"column":3}
```

Resign at `POST /v1/matches/{id}/resign` with `{"action_id":"resign-1"}`.
Successful actions return the acting player's `Observation`.

## Observation

`GET /v1/matches/{id}/observation` returns:

| Field | Meaning |
|---|---|
| `protocol_version`, `observation_version` | Both `"1.0"` |
| `match_id`, `player` | Match identity and your seat, 0 or 1 |
| `phase`, `revision`, `turn` | Lifecycle, state revision, active seat or null |
| `ready` | Two fleet-commitment booleans; no fleet geometry |
| `own_fleet` | Your ship names, lengths, cell coordinates, and sunk flags |
| `incoming_shots` | Events fired at your board |
| `target_board` | 10 row-major rows of 10 cell-state strings |
| `remaining_enemy_ships` | Names not yet publicly sunk, in fleet order |
| `shot_history` | Public ordered events from both players |
| `legal_shot_mask` | 10×10 booleans; true for an unshot cell on your shooting turn |
| `result` | Null, or `{winner, reason}` |

Target cell states are `unknown`, `miss`, `hit` (unresolved), and `sunk`.
Sinking relabels all the disclosed ship cells as `sunk`; no adjacent water is inferred.
Remaining ships are ordered carrier, battleship, cruiser, submarine, destroyer.

Each shot event has `number` (starting at 1), `attacker`, `coordinate`, `outcome`,
`ship`, and `sunk_cells`. `ship` is null and `sunk_cells` empty except for a sinking
event. Older hit events are not rewritten to reveal ship identity later; use the
sinking event or current target board.

The legal mask is all false before play, on the other player's turn, and after
completion. A result reason is `fleet_destroyed`, `resignation`, or `timeout`.
Completion does not reveal the opponent's unshot ships.

## Retry and error semantics

Every mutation requires an `action_id`: 1–128 ASCII letters, digits, underscores,
or hyphens. Generate a fresh UUID for each new logical action, and reuse it only
when retrying that exact action.

- Creation retries with the same ID return the same room and credentials.
- A join retry with the same ID and invitation returns the original second-seat
  credentials. Reusing a consumed invitation with a new ID is rejected.
- Placement, shot, and resignation IDs share a namespace per player. The same ID
  and body replay the original response, without another turn or RNG advance.
- IDs reused with different content or a different operation return `action_conflict`.
- Authenticated engine rejections are cached too. Use a new action ID for a new
  attempt after correcting an action or waiting for your turn.
- A replay returns the original snapshot, which may now be old. Poll observation
  before making a subsequent decision if necessary.

Errors have `{code, message}`. HTTP 400 means invalid fleet geometry, 401 invalid
credentials/invitation, 404 unknown match, 409 state/action conflict, and 422 invalid
request schema. Errors do not echo submitted fleet bodies or credentials. Unknown
body fields and noninteger/out-of-range coordinates are rejected. Business-rule
rejections do not mutate the game; an independently expired deadline still applies.

The SDK retries transport failures and HTTP 502/503/504 with the same action ID
and bounded backoff. It does not retry application errors. It uses request timeouts,
does not follow redirects, and ignores proxy environment variables by default.

## Python SDK and custom policies

```python
from battleship_client import BattleshipClient
from battleship_protocol import Coordinate

with BattleshipClient("http://127.0.0.1:8000") as client:
    room = client.create()
    # Share room.match_id and room.invitation with the other player.
    client.place()  # automatic placement
    observation = client.observe()
    # When observation.turn == observation.player and phase == "playing":
    # observation = client.fire(Coordinate(row=4, column=3))
```

`client.join(match_id, invitation)` attaches the second seat. `client.session` can
be persisted privately and supplied to `BattleshipClient(session=...)` to resume.
Each SDK client represents one seat; use separate instances per player.

```python
from battleship_client.policies import play_policy
from battleship_protocol import Coordinate, Observation

class MyPlayer:
    def choose_shot(self, observation: Observation) -> Coordinate:
        for row, cells in enumerate(observation.legal_shot_mask):
            for column, legal in enumerate(cells):
                if legal:
                    return Coordinate(row=row, column=column)
        raise ValueError("No legal shots")

    # Optional: return five ShipPlacement objects, or None for automatic placement.
    # def choose_placement(self, observation):
    #     return None

# After client.create() or client.join(...):
# result = play_policy(client, MyPlayer())
```

`play_policy` automatically places if no hook exists, polls while waiting, and only
requests shots on the policy's turn. It returns the final observation. Exceptions
from the policy are surfaced; they are not silently replaced with random actions.
The complete executable example is `examples/custom_player.py`.

Fixed network policies are available with the optional `battleship-client[weights]`
extra. Import `WeightPlayer` and `load_weights` from `battleship_client.weights`,
then pass the player to the same `play_policy` runner. The weights module uses
NumPy CPU float64 and has no engine dependency. See
[the weight contract](weight-contract.md) for safe edits import, artifacts, and
command-line examples.

## Trust boundary

Only the server stores both fleets. Clients receive their own fleet, public shot
events, and public readiness/turn information. There is no seed field or hidden
state endpoint. The player distribution contains protocol models, SDK, UI, and
bots; it contains no engine.

This does not isolate hostile programs running as the server's OS user. Containers,
filesystem restrictions, remote TLS deployment, accounts, and isolation of hostile
evaluation programs remain future concerns. The trusted local
[evaluation runner](evaluation.md) preserves the observation boundary. Do not pass
server objects, placement seeds, server credentials, or shared RNG instances into
player policies.

## Independent placement and shooting policies

The existing `PlacementPolicy.choose_placement(observation)` hook can construct a
complete fleet locally and return five `ShipPlacement` objects. `play_policy`
invokes it only for an uncommitted seat, submits the fleet once through the SDK,
and then calls `ShotPolicy.choose_shot` on active shooting turns. A resumed
committed seat skips placement. A policy without the hook retains server automatic
placement.

`ComposedPlayer(shooting, placement)` delegates the two decisions independently.
`create_policy(kind, seed, weights_path, placement_weights_path)` optionally loads a
placement model alongside any supported shooter. Models are validated before
match activity. See the [placement weight contract](placement-weight-contract.md)
for encoding, artifacts, seeding, and CLI examples. HTTP schemas remain version 1.0.
