'''
Regression test for the action normalization/rounding overflow in EV_Charger.step().

When sum(actions) > 1, actions are normalized by dividing by their sum, so they
sum to exactly 1. Rounding each normalized action with round(action, 5) can
round several of them *up*, making the rounded actions sum to slightly more
than 1 (up to +5e-6 per port). current_total_amps then exceeds
max_charge_current by more than the 0.0001 A tolerance and step() raises
"sum of amps ... is higher than max charge current".

Example with 3 ports and max_charge_current=32:
    actions = [0.5, 0.43, 0.43]        # sum = 1.36
    normalized = [0.36764705..., 0.31617647..., 0.31617647...]
    round(., 5) = [0.36765, 0.31618, 0.31618]   # all three rounded UP
    sum = 1.00001  ->  total amps = 32.00032 > 32 + 0.0001

The fix truncates the normalized actions towards zero instead of rounding, so
they can never sum to more than 1.
'''

import pytest

from ev2gym.models.ev_charger import EV_Charger
from ev2gym.models.ev import EV


def make_charger_with_evs(n_ports=3, max_charge_current=32):
    charger = EV_Charger(id=0,
                         connected_bus=0,
                         connected_transformer=0,
                         n_ports=n_ports,
                         max_charge_current=max_charge_current,
                         timescale=5)
    for i in range(n_ports):
        ev = EV(id=i,
                location=0,
                battery_capacity_at_arrival=10,
                time_of_arrival=0,
                time_of_departure=100,
                timescale=5)
        charger.evs_connected[i] = ev
        charger.n_evs_connected += 1
    return charger


@pytest.mark.parametrize('actions', [
    [0.5, 0.43, 0.43],
    [0.71, 0.94, 0.65],
    [0.86, 0.36, 0.73],
])
def test_normalized_actions_do_not_exceed_max_charge_current(actions):
    '''Unequal actions whose normalized values all round up must not raise.'''
    charger = make_charger_with_evs(n_ports=3, max_charge_current=32)

    charger.step(list(actions), charge_price=0.1, discharge_price=0.1)

    assert charger.current_total_amps <= charger.max_charge_current + 0.0001
