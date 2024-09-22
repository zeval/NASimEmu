import gymnasium as gym
import random, logging
import sys
sys.path.insert(1, '/Users/zeval/repositories/thesis/related-work/nasimemu/src')



import nasimemu.env_utils as env_utils

# In this example, a scenario instance is randomly generated from either 'entry_dmz_one_subnet' or 'entry_dmz_two_subnets' on every new episode. Make sure the path to scenarios is correct.
# To use emulation, setup Vagrant and change emulate=True.
env = gym.make('NASimEmu-v0', emulate=False, scenario_name='scenarios/sm_entry_dmz_one_subnet.v2.yaml', disable_env_checker=True)
s = env.reset()

# To see the emulation logs, uncomment the following:
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger('urllib3').setLevel(logging.INFO)

# To see the whole network, use (only in simulation):
# env.render_state()

for _ in range(3):
    actions = env_utils.get_possible_actions(env, s)
    env.render(s)

    # you can convert the observation into a graph format (e.g., for Pytorch Geometric) as:
    # s_graph = env_utils.convert_to_graph(s)

    action = random.choice(actions)
    s, r, done, info = env.step(action)

    print(f"Possible actions: {actions}")

    (action_subnet, action_host), action_id = action
    print(f"Taken action: {action}; subnet_id={action_subnet}, host_id={action_host}, action={env.action_list[action_id]}")
    print(f"reward: {r}, done: {done}\n")
    input()
