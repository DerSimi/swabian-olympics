import pickle
import time
import hockey.hockey_env as h_env


def visualize_game(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"Player: {data['user_names']}")
    print(f"Rounds: {len(data['rounds'])}")

    env = h_env.HockeyEnv(keep_mode=True)

    for i, round_data in enumerate(data["rounds"]):
        print(f"--- Start Round {i+1} ---")

        observations = round_data["observations"]

        env.reset()

        for obs in observations:
            env.set_state(obs)

            env.render()
            time.sleep(0.02)

        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    pkl_file = "41d408f3-9c2c-45a1-96c8-f2f799b2324c.pkl"
    visualize_game(pkl_file)
